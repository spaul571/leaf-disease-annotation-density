"""UADH - Uncertainty-Aware Detection Head for YOLOv11.

MOTIVATION (this is the paper's argument, not decoration).
The v12 campaign established that performance is bounded by ANNOTATION PROTOCOL: Group A
whole-leaf ~0.97 vs Group B lesion ~0.78, a gap of ~0.19 that is 17x its own noise bar and
survives model size, architecture family, imbalance handling, attention and loss changes.
The obvious fix for "small multi-instance lesions" - a P2 stride-4 head - was tried and made
Group B WORSE in two independent environments (T4 -0.029, local -0.054). Diagnosis: P2 quadruples
anchor density (34k vs 8.4k anchors) over exactly the classes carrying the 1,735 conflicting-label
groups, so it adds supervision capacity against contradictory labels and amplifies the noise.

The lesson is that the model needs ROBUSTNESS to label noise, not more resolution. UADH is the
architectural expression of that: a third head branch predicts a per-anchor log-variance s, and
the box loss becomes a heteroscedastic negative log-likelihood

    L_box = mean_fg[ exp(-s) * (1 - IoU) * w  +  0.5 * s * w ]

so the network can declare a box ambiguous (large s) and have its localization error discounted,
while paying a 0.5*s price for doing so. Unlike DFL - whose per-side distribution is trained
toward a hard delta at the label and therefore cannot express doubt about the LABEL itself -
this gives the model an explicit channel for annotation uncertainty.

Design notes:
  * The final conv bias is initialised to 0 => s = 0 => exp(-s) = 1 at step 0, so training starts
    EXACTLY at the baseline objective and any divergence is learned, not an initialisation
    artifact.
  * s is clamped to [-3, 3] (variance ratio ~20x either way) for numerical stability.
  * Uncertainty is used for TRAINING ONLY. Inference is untouched, so mAP is measured with the
    stock decoder and stays directly comparable to every other arm.
  * The unc tensor is handed to the loss through a module-level slot rather than by rewriting
    `get_assigned_targets_and_loss`; BboxLoss.forward already receives `fg_mask`, which is all
    that is needed to select foreground anchors. Patching less = fewer ways to be wrong.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

# slot the patched forward_head writes to and the patched loss reads from, same iteration
_UNC = {"v": None}
_WARNED = {"v": False}
S_CLAMP = 3.0


def _build_branch(det):
    """Create cv4 on a Detect instance. Returns number of params added."""
    chs = [seq[0].conv.in_channels for seq in det.cv2]
    branch = []
    for c in chs:
        c4 = max(16, c // 4)
        head = nn.Sequential(Conv(c, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 1, 1))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        branch.append(head)
    det.cv4 = nn.ModuleList(branch)
    return sum(p.numel() for p in det.cv4.parameters()), chs


_orig_detect_init = Detect.__init__


def _detect_init(self, nc=80, reg_max=16, end2end=False, ch=()):
    """Detect.__init__ + the uncertainty branch.

    CRITICAL: this must be patched at the CLASS level, not attached to an instance.
    `YOLO.train()` REBUILDS the model from `self.model.yaml` and copies weights in, which
    silently discards any module attached to an existing instance - verified 2026-08-11:
    a branch attached before .train() is gone afterwards, and the pre-train smoke test still
    passes because it runs on the pre-rebuild model. Patching __init__ means the rebuilt
    Detect constructs the branch itself.
    """
    _orig_detect_init(self, nc, reg_max, end2end, ch)
    if ch:
        _build_branch(self)


def attach_uadh(model, verbose=True):
    """Ensure the branch exists on an already-built model (the class patch covers rebuilds)."""
    det = model.model.model[-1]
    assert isinstance(det, Detect), f"last layer is {type(det).__name__}, not Detect"
    if hasattr(det, "cv4"):
        if verbose:
            added = sum(p.numel() for p in det.cv4.parameters())
            total = sum(p.numel() for p in model.model.parameters())
            print(f"[uadh] branch present | +{added:,} params -> {total/1e6:.3f}M "
                  f"({100*added/total:.1f}% increase)")
        return model

    chs = [seq[0].conv.in_channels for seq in det.cv2]  # per-level input channels
    branch = []
    for c in chs:
        c4 = max(16, c // 4)
        head = nn.Sequential(Conv(c, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 1, 1))
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)   # s = 0 at init => identical objective to baseline
        branch.append(head)
    det.cv4 = nn.ModuleList(branch).to(next(det.parameters()).device)

    added = sum(p.numel() for p in det.cv4.parameters())
    total = sum(p.numel() for p in model.model.parameters())
    if verbose:
        print(f"[uadh] branch attached to {len(chs)} levels (ch={chs}) | "
              f"+{added:,} params -> {total/1e6:.3f}M total ({100*added/total:.1f}% increase)")
    return model


def _forward_head(self, x, box_head=None, cls_head=None):
    """Stock forward_head plus the uncertainty branch (see ultralytics 8.4.x source)."""
    if box_head is None or cls_head is None:  # fused inference
        return {}
    bs = x[0].shape[0]
    boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
    scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
    out = {"boxes": boxes, "scores": scores, "feats": x}
    # only the one2many pass owns cv4; guard so the end2end path can't double-write the slot
    if hasattr(self, "cv4") and box_head is self.cv2:
        # Run the branch outside autocast for numerical safety: exp(-s) in the NLL is the one
        # place fp16 can bite, and the branch is <1% of runtime so fp32 costs nothing.
        # HONEST NOTE: an earlier session concluded "AMP kills this branch" from a coco8 test
        # where the final conv stayed at exactly 0. That test was CONFOUNDED - with 8 images at
        # batch 2, ultralytics' gradient accumulation (nbs 64 / batch = 32) meant ~1 optimizer
        # step in the whole run, so nothing could move regardless of AMP. On the real dataset
        # the branch trains fine. This guard is defensive, not a proven fix.
        # Cast to the BRANCH's own dtype, not unconditionally to fp32: during training the cv4
        # params are fp32 (so disabling autocast gives a genuine fp32 branch), while
        # ultralytics' AMP self-check and fused inference run a .half() model, where forcing
        # fp32 input would raise "Input type (float) and bias type (c10::Half)".
        p_dtype = next(self.cv4.parameters()).dtype
        with torch.autocast(device_type=x[0].device.type, enabled=False):
            unc = torch.cat([self.cv4[i](x[i].to(p_dtype)).view(bs, 1, -1)
                             for i in range(self.nl)], dim=-1)
            unc = unc.squeeze(1).float().clamp(-S_CLAMP, S_CLAMP)   # (bs, num_anchors)
        _UNC["v"] = unc
        out["unc"] = unc
    return out


def _bbox_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                  target_scores, target_scores_sum, fg_mask, imgsz, stride):
    """Stock BboxLoss.forward with ONLY the IoU term replaced by the heteroscedastic NLL.

    DFL branch is copied verbatim from ultralytics 8.4.115 - do not 'improve' it.
    """
    weight = target_scores[fg_mask].sum(-1, keepdim=True)
    iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)

    s = _UNC["v"]
    if s is None or s.shape != fg_mask.shape or s.device != fg_mask.device:  # noqa: E501
        # LOUD, not silent: a quiet fallback here would turn UADH into a no-op that still
        # reports as "the UADH arm" - exactly the failure mode that hid the discarded branch.
        if not _WARNED["v"]:
            _WARNED["v"] = True
            print(f"[uadh] WARNING falling back to stock CIoU - unc slot "
                  f"{'empty' if s is None else f'{tuple(s.shape)}@{s.device} vs fg_mask '
                     f'{tuple(fg_mask.shape)}@{fg_mask.device}'}", flush=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
    else:
        # fp32 for the same overflow reason as the branch itself
        with torch.autocast(device_type=iou.device.type, enabled=False):
            s_fg = s[fg_mask].unsqueeze(-1).float()                    # (n_fg, 1)
            nll = torch.exp(-s_fg) * (1.0 - iou.float()) + 0.5 * s_fg
            loss_iou = (nll * weight.float()).sum() / target_scores_sum

    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def patch_uadh():
    """Install the head and loss patches (idempotent)."""
    Detect.__init__ = _detect_init          # survives the rebuild inside YOLO.train()
    Detect.forward_head = _forward_head
    BboxLoss.forward = _bbox_forward
    _WARNED["v"] = False
    print("[uadh] patched Detect.__init__ (+cv4), Detect.forward_head (+unc), "
          "BboxLoss.forward (heteroscedastic NLL)")


def assert_uadh_alive(model, where=""):
    """Fail loudly if the branch is missing - call AFTER training, not just before."""
    det = model.model.model[-1]
    ok = hasattr(det, "cv4")
    print(f"[uadh] branch present {where}: {ok}")
    assert ok, f"UADH branch missing {where} - the arm would be a plain baseline"
    return ok
