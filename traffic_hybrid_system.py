"""
Hệ thống kết hợp 2 model:
  - Vehicle.pt      : YOLO detect (car / motocycle / truck / bus — tên lớp lấy từ model.names)
  - ViTs+CBAM.pt    : YOLO segmentation (person / head / helmet — tên lớp lấy từ model.names)

Chức năng:
  1) Vượt đèn đỏ: ROI màu đèn (HSV) + vạch dừng ảo + tracking xe cắt vạch khi đèn đỏ
  2) Không đội mũ: trên vùng xe máy (mở rộng bbox), chạy seg; nếu có head/person mà không đủ mặt nạ helmet → vi phạm

Chạy:
  python traffic_hybrid_system.py --video path/to/video.mp4 --out out.mp4
  python traffic_hybrid_system.py --video path/to/video.mp4 --show

Yêu cầu: đặt weights tại models/Vehicle.pt và models/ViTs+CBAM.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
# FIX: chuyển toàn bộ typing imports lên trước để tránh NameError
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Checkpoint seg có thể chứa BCEDiceLoss tùy chỉnh — cần patch trước khi load weight
try:
    import patch_ultralytics_custom_loss  # noqa: F401
except ImportError:
    pass

try:
    from ultralytics import YOLO
except ImportError:
    print("Cần cài: pip install ultralytics torch")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------


@dataclass
class HybridConfig:
    """Tham số cần tinh chỉnh theo từng camera / video."""

    vehicle_weights: Path = Path("models/Vehicle.pt")
    seg_weights: Path = Path("models/ViTs+CBAM.pt")

    conf_vehicle: float = 0.35
    iou_vehicle: float = 0.5
    conf_seg: float = 0.35

    # Vạch dừng: tỉ lệ theo chiều cao khung hình (0 = trên, 1 = dưới)
    stop_line_y_ratio: float = 0.78

    # ROI đèn giao thông (tỉ lệ trên khung hình): x1,y1,x2,y2 trong [0,1]
    traffic_light_roi_ratio: Tuple[float, float, float, float] = (0.25, 0.0, 0.75, 0.22)

    # HSV đỏ (có thể chỉnh theo camera)
    red_light_min_ratio: float = 0.14

    # Tên lớp xe máy (thử theo thứ tự)
    motorcycle_name_candidates: Tuple[str, ...] = (
        "motocycle",
        "motorcycle",
        "motorbike",
        "motor",
    )

    # Ngưỡng diện tích mặt nạ để coi là có helmet/head.
    helmet_area_ratio_min: float = 0.0008
    head_area_ratio_min: float = 0.0005

    # Mở rộng bbox xe máy trước khi crop cho seg
    moto_crop_pad: float = 0.45

    tracker: str = "bytetrack.yaml"

    # Debug: vẽ segmentation overlay trực tiếp lên khung xe máy để bạn kiểm chứng helmet/head
    debug_overlay_seg: bool = True
    debug_show_stop_y: bool = True

    # Debug: in thống kê seg để biết model có detect class nào không
    debug_log_seg: bool = True
    debug_log_seg_max_per_frame: int = 3
    seg_imgsz: int = 640
    # Fallback nhẹ hơn nếu 0 detection (không để quá thấp tránh noise)
    seg_fallback_extra: Tuple[Tuple[float, int], ...] = (
        (0.20, 960),
        (0.10, 1280),
    )
    seg_max_det: int = 100

    # Bỏ qua mask có diện tích bbox > X% frame (tránh mask phủ toàn khung)
    max_mask_area_ratio: float = 0.20

    # Tỉ lệ overlap tối thiểu giữa seg box và moto crop để tính là "trong vùng xe máy"
    seg_overlap_threshold: float = 0.3

    # ── Chế độ chỉ chạy ViTs+CBAM, bỏ qua Vehicle detection ──────────────────
    # Bật = True để test riêng model seg mà không cần Vehicle.pt
    seg_only_mode: bool = False


@dataclass
class FrameViolations:
    red_light_ids: Set[int] = field(default_factory=set)
    no_helmet_moto_indices: List[int] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_tight_bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Từ mask nhị phân (0/1), trả về bbox tight nhất chứa mask."""
    ys, xs = np.where(mask > 0.5)
    if len(xs) == 0 or len(ys) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _norm_name(s: str) -> str:
    return s.strip().lower()


def _find_class_ids(names: Dict[int, str], keywords: Tuple[str, ...]) -> List[int]:
    out = []
    for i, raw in names.items():
        n = _norm_name(str(raw))
        for k in keywords:
            kl = k.lower()
            if kl in n or n == kl:
                out.append(int(i))
                break
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Đèn đỏ (HSV)
# ---------------------------------------------------------------------------


def is_red_light_in_roi(frame: np.ndarray, roi_xyxy: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = roi_xyxy
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([12, 255, 255])
    lower_red2 = np.array([165, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2),
    )
    return float(np.sum(mask > 0)) / float(mask.size) > 0.14


def ratio_to_xyxy(
    wh: Tuple[int, int], r: Tuple[float, float, float, float]
) -> Tuple[int, int, int, int]:
    w, h = wh
    x1, y1, x2, y2 = r
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def expand_xyxy(
    xyxy: np.ndarray, w: int, h: int, pad: float
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, xyxy)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw
    y1 -= pad * bh
    x2 += pad * bw
    y2 += pad * bh
    return (
        max(0, int(x1)),
        max(0, int(y1)),
        min(w - 1, int(x2)),
        min(h - 1, int(y2)),
    )


def box_area_norm_in_crop(
    mask_hw: np.ndarray, crop_shape: Tuple[int, int]
) -> float:
    """Tỉ lệ pixel mask trên diện tích crop."""
    ch, cw = crop_shape
    if ch <= 0 or cw <= 0:
        return 0.0
    if mask_hw.shape[0] != ch or mask_hw.shape[1] != cw:
        mask_hw = cv2.resize(
            mask_hw.astype(np.float32), (cw, ch), interpolation=cv2.INTER_NEAREST
        )
    return float(np.sum(mask_hw > 0.5)) / float(ch * cw)


# ---------------------------------------------------------------------------
# Hệ thống chính
# ---------------------------------------------------------------------------


class TrafficHybridSystem:
    def __init__(self, cfg: Optional[HybridConfig] = None):
        self.cfg = cfg or HybridConfig()
        vw = self.cfg.vehicle_weights
        sw = self.cfg.seg_weights
        if not vw.is_file():
            raise FileNotFoundError(f"Không thấy {vw} — đặt weight detect vào đây.")
        if not sw.is_file():
            raise FileNotFoundError(f"Không thấy {sw} — đặt weight segmentation vào đây.")

        self.det = YOLO(str(vw))
        self.seg = YOLO(str(sw))
        self.veh_names: Dict[int, str] = self.det.names
        self.seg_names: Dict[int, str] = self.seg.names

        _seg_task = getattr(self.seg, "task", None)
        if _seg_task is None:
            try:
                _seg_task = getattr(self.seg.model, "task", None)
            except Exception:
                _seg_task = None
        print(f"[ViTs+CBAM.pt] task={_seg_task!r}")
        if _seg_task and str(_seg_task).lower() in ("detect", "detection"):
            print(
                ">>> LOI: Weight nay la YOLO DETECT, KHONG PHAI SEGMENTATION. "
                "Khong co masks. Can file .pt train tu `yolo segment train` / model seg.",
            )

        if self.cfg.debug_overlay_seg:
            print("Vehicle class names (Vehicle.pt):", self.veh_names)
            print("Seg class names (ViTs+CBAM.pt):", self.seg_names)

        self._moto_ids: Set[int] = set()
        for cand in self.cfg.motorcycle_name_candidates:
            for i, n in self.veh_names.items():
                if cand.lower() in str(n).lower():
                    self._moto_ids.add(int(i))
        if not self._moto_ids:
            print(
                "Canh bao: khong map duoc lop xe may trong Vehicle.pt — kiem tra model.names"
                " va dat motorcycle_name_candidates trong HybridConfig.",
            )

        # Cache class ids cho seg để không tính lại mỗi frame
        self._ids_helmet: List[int] = _find_class_ids(
            self.seg_names,
            ("helmet", "no_helmet", "nohelmet", "no-helmet", "mu", "mũ", "khong_mu", "khongmu"),
        )
        self._ids_head: List[int] = _find_class_ids(
            self.seg_names, ("head", "dau", "đầu")
        )
        self._ids_person: List[int] = _find_class_ids(
            self.seg_names, ("person", "nguoi", "người", "people")
        )

        if self.cfg.debug_overlay_seg:
            print(f"  helmet class ids : {self._ids_helmet}")
            print(f"  head   class ids : {self._ids_head}")
            print(f"  person class ids : {self._ids_person}")

        self._frame_idx: int = 0
        self._seg_warned_empty: bool = False

    def _run_segmentation(self, frame: np.ndarray):
        """
        Chạy seg với chuỗi fallback (conf thấp dần + imgsz lớn hơn) nếu 0 detection.
        """
        cfg = self.cfg
        attempts = [(cfg.conf_seg, cfg.seg_imgsz)] + list(cfg.seg_fallback_extra)
        last = None
        for conf, imgsz in attempts:
            kwargs = dict(
                conf=conf,
                imgsz=imgsz,
                max_det=cfg.seg_max_det,
                verbose=False,
                retina_masks=True,
            )
            try:
                last = self.seg(frame, **kwargs)[0]
            except TypeError:
                kwargs.pop("retina_masks", None)
                last = self.seg(frame, **kwargs)[0]

            n = 0 if last.boxes is None else len(last.boxes)
            if n > 0:
                if not self._seg_warned_empty and self.cfg.debug_log_seg:
                    print(f"[seg] OK: conf={conf} imgsz={imgsz} dets={n}")
                self._seg_warned_empty = False
                return last

        if not self._seg_warned_empty and self.cfg.debug_log_seg:
            print(
                "[seg] Van 0 detection sau fallback. Kiem tra: (1) weight co phai seg khong, "
                "(2) video/domain khac train, (3) giam conf_seg / tang seg_imgsz trong HybridConfig.",
            )
            self._seg_warned_empty = True
        return last

    def _is_motorcycle(self, cls_id: int) -> bool:
        return int(cls_id) in self._moto_ids

    def _check_helmet_in_region(
        self,
        mx1: int, my1: int, mx2: int, my2: int,
        seg_boxes: np.ndarray,
        seg_clss: np.ndarray,
        seg_masks,           # có thể là None
        frame_shape: Tuple[int, int],
    ) -> Tuple[bool, bool, str]:
        """
        Kiểm tra trong vùng moto crop có helmet / head / person không.

        Returns:
            has_helmet (bool)
            has_head_or_person (bool)
            debug_reason (str)
        """
        cfg = self.cfg
        has_helmet = False
        has_head_or_person = False

        # FIX: kiểm tra seg_boxes rỗng an toàn (tránh lỗi khi len() gọi trên None hoặc list rỗng)
        if seg_boxes is None or len(seg_boxes) == 0:
            return False, False, "no_seg_dets"

        fh, fw = frame_shape[:2]
        crop_area = max(1, (mx2 - mx1) * (my2 - my1))

        for j in range(len(seg_boxes)):
            sc = int(seg_clss[j])

            # Bỏ qua class không liên quan
            if sc not in self._ids_helmet and sc not in self._ids_head and sc not in self._ids_person:
                continue

            sb = seg_boxes[j]  # [x1, y1, x2, y2] toàn frame
            sx1, sy1, sx2, sy2 = float(sb[0]), float(sb[1]), float(sb[2]), float(sb[3])

            # Giao nhau với vùng moto crop
            ix1 = max(mx1, sx1)
            iy1 = max(my1, sy1)
            ix2 = min(mx2, sx2)
            iy2 = min(my2, sy2)

            if ix2 <= ix1 or iy2 <= iy1:
                # Không giao nhau
                continue

            inter_area = (ix2 - ix1) * (iy2 - iy1)
            sbox_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
            overlap_ratio = inter_area / sbox_area

            # FIX: nếu có mask thì dùng mask area thay vì bbox area để chính xác hơn
            if seg_masks is not None and j < len(seg_masks):
                try:
                    m = seg_masks[j]
                    if hasattr(m, "cpu"):
                        m = m.cpu().numpy()
                    m = np.asarray(m, dtype=np.float32)

                    # Mask từ YOLO seg thường có shape (H_mask, W_mask) — cần resize về frame
                    if m.shape[0] != fh or m.shape[1] != fw:
                        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)

                    # Đếm pixel mask nằm trong vùng moto crop
                    mask_in_crop = m[my1:my2, mx1:mx2]
                    mask_pixels_in_crop = float(np.sum(mask_in_crop > 0.5))
                    total_mask_pixels = max(1.0, float(np.sum(m > 0.5)))
                    overlap_ratio = mask_pixels_in_crop / total_mask_pixels

                    # Cũng kiểm tra diện tích tuyệt đối trong crop so với crop
                    area_ratio_in_crop = mask_pixels_in_crop / crop_area
                    if sc in self._ids_helmet and area_ratio_in_crop < cfg.helmet_area_ratio_min:
                        continue
                    if (sc in self._ids_head or sc in self._ids_person) and area_ratio_in_crop < cfg.head_area_ratio_min:
                        continue

                except Exception as e:
                    # Fallback về bbox nếu mask lỗi
                    pass

            if overlap_ratio < cfg.seg_overlap_threshold:
                continue

            if sc in self._ids_helmet:
                has_helmet = True
            elif sc in self._ids_head or sc in self._ids_person:
                has_head_or_person = True

        reason = "ok"
        if has_helmet:
            reason = "co_mu"
        elif not has_head_or_person:
            reason = "khong_ro_head"
        else:
            reason = "khong_mu"

        return has_helmet, has_head_or_person, reason

    def process_frame(
        self, frame: np.ndarray, track_ids: bool = True
    ) -> Tuple[np.ndarray, FrameViolations]:
        self._frame_idx += 1
        h, w = frame.shape[:2]
        cfg = self.cfg
        fv = FrameViolations()

        stop_y = int(h * cfg.stop_line_y_ratio)
        light_roi = ratio_to_xyxy((w, h), cfg.traffic_light_roi_ratio)
        red = is_red_light_in_roi(frame, light_roi)

        # --- Detection / Tracking xe (bỏ qua khi seg_only_mode) ---
        if cfg.seg_only_mode:
            det_results = None
        elif track_ids:
            det_results = self.det.track(
                frame,
                persist=True,
                conf=cfg.conf_vehicle,
                iou=cfg.iou_vehicle,
                tracker=cfg.tracker,
                verbose=False,
            )
        else:
            det_results = self.det(
                frame, conf=cfg.conf_vehicle, iou=cfg.iou_vehicle, verbose=False
            )

        # --- Segmentation toàn khung (fallback conf/imgsz nếu 0 det) ---
        seg_result = self._run_segmentation(frame)

        # FIX: lấy boxes/cls/masks an toàn, trả về array rỗng thay vì None
        if seg_result.boxes is not None and len(seg_result.boxes) > 0:
            seg_boxes = seg_result.boxes.xyxy.cpu().numpy()
            seg_clss = seg_result.boxes.cls.cpu().numpy().astype(int)
        else:
            seg_boxes = np.empty((0, 4), dtype=np.float32)
            seg_clss = np.empty((0,), dtype=int)

        # FIX: lấy masks an toàn
        seg_masks = None
        if seg_result.masks is not None:
            try:
                seg_masks = seg_result.masks.data
                if hasattr(seg_masks, "cpu"):
                    seg_masks = seg_masks.cpu().numpy()
                seg_masks = np.asarray(seg_masks)
            except Exception:
                seg_masks = None

        if cfg.debug_log_seg:
            cls_names = [self.seg_names.get(int(i), str(i)) for i in seg_clss]
            has_masks_str = f"masks={seg_masks is not None and len(seg_masks) > 0}"
            print(
                f"[frame {self._frame_idx}] seg dets={len(seg_boxes)} {has_masks_str}"
                f" cls={cls_names[:cfg.debug_log_seg_max_per_frame * 3]}"
            )

        annotated = frame.copy()
        frame_area = h * w

        # Overlay seg: contour mask + nhãn nhỏ
        if cfg.debug_overlay_seg and seg_masks is not None and len(seg_boxes) > 0:
            _cls_colors = {
                "helmet": (0, 255, 0),    # xanh lá
                "head":   (0, 165, 255),  # cam
                "person": (255, 0, 255),  # tím
            }
            _relevant_ids = set(self._ids_helmet) | set(self._ids_head) | set(self._ids_person)

            for j in range(min(len(seg_boxes), len(seg_masks))):
                sc = int(seg_clss[j]) if j < len(seg_clss) else -1
                sname = self.seg_names.get(sc, str(sc)).lower()

                # seg_only_mode: hiện tất cả class để debug
                # chế độ thường: chỉ helmet/head/person
                if not cfg.seg_only_mode and sc not in _relevant_ids:
                    continue

                # ── Lấy mask binary (instance segmentation pixel-level) ──────
                try:
                    m = seg_masks[j]
                    m_np = np.asarray(m, dtype=np.float32)
                    # retina_masks=True → mask đã full-res; nếu không thì resize
                    if m_np.shape[0] != h or m_np.shape[1] != w:
                        m_np = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    m_bin = (m_np > 0.5).astype(np.uint8)
                except Exception:
                    continue

                # Lọc mask quá lớn dựa trên SỐ PIXEL THỰC (không phải bbox)
                mask_pixel_ratio = float(m_bin.sum()) / frame_area
                if mask_pixel_ratio > cfg.max_mask_area_ratio:
                    continue
                # Bỏ qua mask quá nhỏ (noise)
                if mask_pixel_ratio < 0.0002:
                    continue

                # Màu theo class
                if sc in self._ids_helmet:
                    color = _cls_colors["helmet"]
                elif sc in self._ids_head:
                    color = _cls_colors["head"]
                elif sc in self._ids_person:
                    color = _cls_colors["person"]
                else:
                    # Màu determin theo class id cho debug
                    _h = (sc * 47 + 30) % 180
                    hsv_c = np.uint8([[[_h, 220, 210]]])
                    color = tuple(int(x) for x in cv2.cvtColor(hsv_c, cv2.COLOR_HSV2BGR)[0][0])

                # Vẽ fill nhạt + contour
                m_draw = m_bin * 255
                overlay_mask = np.zeros_like(annotated)
                overlay_mask[m_bin > 0] = color
                annotated = cv2.addWeighted(annotated, 1.0, overlay_mask, 0.25, 0)
                contours, _ = cv2.findContours(m_draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, color, 2)

                # Bbox + nhãn nhỏ
                bx1, by1, bx2, by2 = map(int, seg_boxes[j])
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(annotated, sname, (bx1, max(0, by1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Vẽ vạch dừng + ROI đèn
        cv2.line(annotated, (0, stop_y), (w, stop_y), (0, 255, 255), 3)
        cv2.putText(
            annotated, "STOP LINE", (10, stop_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        if cfg.debug_show_stop_y:
            cv2.putText(
                annotated, f"y={stop_y}", (10, stop_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )
        lx1, ly1, lx2, ly2 = light_roi
        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), (255, 200, 0), 2)
        status = "DEN DO" if red else "KHONG DO (hoac ROI can chinh)"
        cv2.putText(
            annotated, status, (lx1, ly1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 255) if red else (0, 180, 0), 2,
        )

        r0 = det_results[0] if det_results is not None else None
        if r0 is None or r0.boxes is None or len(r0.boxes) == 0:
            return annotated, fv

        boxes = r0.boxes.xyxy.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        ids = None
        if r0.boxes.id is not None:
            ids = r0.boxes.id.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            xyxy = boxes[i]
            c = clss[i]
            tid = int(ids[i]) if ids is not None else i
            label = self.veh_names.get(int(c), str(c))
            x1, y1, x2, y2 = map(int, xyxy)

            color = (0, 255, 0)
            thick = 2

            # --- Vượt đèn đỏ ---
            veh_labels_lower = {"car", "bus", "truck", "motocycle", "motorcycle", "motorbike"}
            is_veh = any(v in label.lower() for v in veh_labels_lower) or self._is_motorcycle(c)
            if red and y2 > stop_y and is_veh:
                fv.red_light_ids.add(tid)
                color = (0, 0, 255)
                thick = 3
                fv.messages.append(f"ID {tid}: vuot den do")
                cv2.putText(
                    annotated, f"VUOT DEN! id={tid}",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            # --- Không đội mũ (chỉ xe máy) ---
            if self._is_motorcycle(c):
                mx1, my1, mx2, my2 = expand_xyxy(xyxy, w, h, cfg.moto_crop_pad)

                has_helmet, has_head_or_person, reason = self._check_helmet_in_region(
                    mx1, my1, mx2, my2,
                    seg_boxes, seg_clss, seg_masks,
                    frame.shape,
                )

                if cfg.debug_overlay_seg:
                    cv2.rectangle(annotated, (mx1, my1), (mx2, my2), (255, 0, 255), 1)

                # FIX: vi phạm chỉ khi CHẮC CHẮN có người/đầu nhưng không có mũ
                # Nếu không detect được gì (no_seg_dets / khong_ro_head) → KHÔNG phạt
                bad = (not has_helmet) and has_head_or_person

                if bad:
                    fv.no_helmet_moto_indices.append(i)
                    color = (0, 0, 255)
                    thick = 3
                    fv.messages.append(f"Moto #{i} id={tid}: khong mu ({reason})")
                    cv2.putText(
                        annotated, "KHONG MU!",
                        (x1, y2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )
                elif cfg.debug_overlay_seg:
                    # Hiện lý do không phạt để debug
                    cv2.putText(
                        annotated, f"[{reason}]",
                        (mx1, my1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 0), 1,
                    )

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thick)
            cv2.putText(
                annotated, f"{label} {tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
            )

        return annotated, fv


# ---------------------------------------------------------------------------
# Chạy video
# ---------------------------------------------------------------------------


def run_video(
    video_path: str,
    out_path: Optional[str],
    show: bool,
    cfg: Optional[HybridConfig] = None,
) -> None:  # FIX: bỏ khai báo cfg thừa, thêm dấu ) đúng vị trí
    sys_ = TrafficHybridSystem(cfg or HybridConfig())
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_i = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_i += 1
            ann, viol = sys_.process_frame(frame)
            if writer:
                writer.write(ann)
            if show:
                cv2.imshow("TrafficHybrid", ann)
                if cv2.waitKey(1) == 27:
                    break
            if frame_i % 30 == 0 and viol.messages:
                print(f"[frame {frame_i}]", "; ".join(viol.messages[-3:]))
    finally:
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    print(f"Xong. {frame_i} frame." + (f" Da luu: {out_path}" if out_path else ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Vehicle + ViTs/CBAM seg — den do + khong mu")
    p.add_argument("--video", type=str, required=True, help="Đường dẫn video đầu vào")
    p.add_argument("--out", type=str, default=None, help="Video đầu ra (mp4)")
    p.add_argument("--show", action="store_true", help="Hiển thị cửa sổ")
    p.add_argument("--vehicle", type=str, default="models/Vehicle.pt")
    p.add_argument("--seg", type=str, default="models/ViTs+CBAM.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = HybridConfig(
        vehicle_weights=Path(args.vehicle),
        seg_weights=Path(args.seg),
    )
    run_video(args.video, args.out, args.show, cfg)