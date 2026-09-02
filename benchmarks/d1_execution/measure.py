"""
D1 measurement collection — clean rerun.

Audit + measurement only. This module NEVER modifies depth_perception_engine
production code or behaviour. Where a per-stage timing is needed it either

  (a) reads the pipeline's OWN already-shipped DEBUG-level stage
      instrumentation (pipeline.py emits "<Stage> stage: %.2f ms" for every
      geometry/temporal stage) by attaching a logging handler, or

  (b) re-drives an already-shipped stage function with the EXACT arguments
      pipeline.py itself passes it, outside the pipeline, for the stages
      pipeline.py does not itself log (rectify / grayscale / SGBM / depth /
      shadow+ramp masks / scene interpretation / threat assessment /
      result + GeometryFrame assembly).

No production file is edited, no algorithm semantics are changed, and no
threads / queues / futures / multiprocessing / async / CUDA are introduced.
"""

import gc
import json
import logging
import os
import platform
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import depth_perception_engine as dpe  # noqa: E402
from depth_perception_engine.core import (  # noqa: E402
    DepthPerceptionPipeline,
    GeometryFrame,
    StereoObservation,
)
from depth_perception_engine.frames import FrameId  # noqa: E402

from benchmarks.d1_execution import fixtures as F  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# helpers
# =====================================================================
def stats_ms(samples: List[float]) -> Dict[str, float]:
    a = np.asarray(samples, dtype=np.float64)
    mean = float(a.mean())
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "stddev_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "fps_from_mean": float(1000.0 / mean) if mean > 0 else None,
        "fps_from_median": float(1000.0 / float(np.median(a))) if np.median(a) > 0 else None,
    }


def rss_mb() -> float:
    with open("/proc/self/statm", "r") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)


def jsonable(x: Any) -> Any:
    if isinstance(x, (str, bool, int)) or x is None:
        return x
    if isinstance(x, float):
        return x if np.isfinite(x) else str(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else str(v)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return {"__ndarray__": True, "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return str(x)


class StageLogCapture(logging.Handler):
    """Reads pipeline.py's OWN existing DEBUG stage instrumentation.

    pipeline.py already emits, per frame, messages of the exact form
    "<Something> stage: 1.23 ms, ..." (plus "Body-frame transform stage",
    "Rotation compensation stage"). Nothing is added to production code to
    make this work — the handler simply listens.
    """

    _PAT = re.compile(r"^(?P<stage>.+?) stage: (?P<ms>[0-9.]+) ms")

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.samples: Dict[str, List[float]] = defaultdict(list)
        self.enabled = False

    def emit(self, record):
        if not self.enabled:
            return
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = self._PAT.match(msg)
        if m:
            self.samples[m.group("stage").strip()].append(float(m.group("ms")))

    def reset(self):
        self.samples = defaultdict(list)


_PIPELINE_LOGGER = logging.getLogger("depth_perception_engine.pipeline.pipeline")


def observation(left, right, ts=None, hints=None, hint=None, frame_id=None, calibration=None) -> StereoObservation:
    return StereoObservation(
        left_image=left,
        right_image=right,
        left_timestamp=ts,
        right_timestamp=None,
        calibration=calibration,
        frame_id=frame_id,
        motion_hint=hint,
        motion_hints=hints,
    )


# =====================================================================
# 0. Environment
# =====================================================================
def section_environment() -> Dict[str, Any]:
    import subprocess

    def sh(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as exc:  # pragma: no cover - diagnostic only
            return f"<unavailable: {exc}>"

    cal = F.calibration()
    return {
        "package_version": dpe.__version__,
        "git_branch": sh("git branch --show-current"),
        "git_head": sh("git rev-parse HEAD"),
        "git_status_short": sh("git status --short"),
        "git_tags": sh("git tag --list").splitlines(),
        "git_describe": sh("git describe --always --tags"),
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "opencv_num_threads": int(cv2.getNumThreads()),
        "opencv_num_cpus": int(cv2.getNumberOfCPUs()),
        "opencv_use_optimized": bool(cv2.useOptimized()),
        "os_cpu_count": os.cpu_count(),
        "cpu_model": sh("lscpu | grep 'Model name' | head -1 | cut -d: -f2 | xargs"),
        "cpu_cores": sh("lscpu | grep '^Core(s) per socket' | cut -d: -f2 | xargs"),
        "cpu_threads_per_core": sh("lscpu | grep '^Thread(s) per core' | cut -d: -f2 | xargs"),
        "ram_total": sh("free -h | awk 'NR==2{print $2}'"),
        "ram_available": sh("free -h | awk 'NR==2{print $7}'"),
        "kernel": platform.platform(),
        "os_release": sh("grep PRETTY_NAME /etc/os-release | cut -d'\"' -f2"),
        "calibration_fixture": F.CALIBRATION_PATH,
        "calibration_image_size": list(cal.image_size),
        "qualified_config": {
            k: jsonable(getattr(F.qualified_config(), k))
            for k in sorted(F.qualified_config().__slots__)
        },
    }


# =====================================================================
# 3/4. StereoObservation contract + observation identity
# =====================================================================
def section_identity() -> Dict[str, Any]:
    cal = F.calibration()
    pipeline = DepthPerceptionPipeline(
        F.qualified_config(), cal, rectify=True, body_T_camera_left=F.body_transform()
    )
    left, right = F.scene_pair()

    obs = observation(left, right, ts=100.0, frame_id="observation-X", hints=None)
    gf = pipeline.process_geometry_frame(obs)

    # Where, if anywhere, does the literal string "observation-X" appear?
    def scan_for(token: str, frame: GeometryFrame) -> List[str]:
        hits = []

        def walk(obj, path, depth=0):
            if depth > 4:
                return
            if isinstance(obj, str):
                if token in obj:
                    hits.append(path)
                return
            if isinstance(obj, (int, float, bool, type(None), np.ndarray)):
                return
            if isinstance(obj, dict):
                for k, v in list(obj.items())[:64]:
                    walk(v, f"{path}[{k!r}]", depth + 1)
                return
            if isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj[:64]):
                    walk(v, f"{path}[{i}]", depth + 1)
                return
            slots = getattr(type(obj), "__slots__", None)
            names = list(slots) if slots else [a for a in dir(obj) if not a.startswith("_")]
            for name in names:
                try:
                    v = getattr(obj, name)
                except Exception:
                    continue
                if callable(v):
                    continue
                walk(v, f"{path}.{name}", depth + 1)

        walk(frame, "GeometryFrame")
        return hits

    # Is StereoObservation.frame_id read anywhere in production source?
    src_root = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")
    reads = []
    for dirpath, _dirs, files in os.walk(src_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if "observation.frame_id" in line or "obs.frame_id" in line:
                        reads.append(f"{os.path.relpath(p, _REPO_ROOT)}:{i}: {line.strip()}")

    nested_frame_ids = {
        "GeometryFrame.frame_id": gf.frame_id,
        "GeometryFrame.geometry.frame_id": gf.geometry.frame_id if gf.geometry is not None else None,
        "GeometryFrame.geometry_body.frame_id": gf.geometry_body.frame_id if gf.geometry_body is not None else None,
        "GeometryFrame.obstacle_cloud.frame_id": gf.obstacle_cloud.frame_id if gf.obstacle_cloud is not None else None,
        "GeometryFrame.free_space_rays.frame_id": gf.free_space_rays.frame_id if gf.free_space_rays is not None else None,
        "GeometryFrame.surface_evidence[0].frame_id": (
            gf.surface_evidence[0].frame_id if gf.surface_evidence else None
        ),
        "GeometryFrame.boundary_evidence[0].frame_id": (
            gf.boundary_evidence[0].frame_id if gf.boundary_evidence else None
        ),
        "GeometryFrame.clearance_evidence[0].frame_id": (
            gf.clearance_evidence[0].frame_id if gf.clearance_evidence else None
        ),
        "GeometryFrame.region_evidence['TL'].frame_id": (
            gf.region_evidence["TL"].frame_id if gf.region_evidence and "TL" in gf.region_evidence else None
        ),
    }

    # Does the legacy DepthPerceptionResult carry it either?
    res = pipeline.process_observation(observation(left, right, ts=101.0, frame_id="observation-Y"))
    legacy_fields_with_token = [
        f for f in res.__slots__
        if isinstance(getattr(res, f, None), str) and "observation-Y" in getattr(res, f)
    ]

    return {
        "stereo_observation_fields": [
            {"name": "left_image", "type": "np.ndarray", "required": True,
             "semantics": "left stereo image, BGR or grayscale",
             "ownership": "BY REFERENCE, never copied by DPE",
             "read_by_dpe": True, "consumed_at": "process_observation -> rectify/gray/SGBM",
             "propagates_to_geometry_frame": "indirectly (disparity/depth rasters derived from it)"},
            {"name": "right_image", "type": "np.ndarray", "required": True,
             "semantics": "right stereo image", "ownership": "BY REFERENCE, never copied",
             "read_by_dpe": True, "consumed_at": "process_observation -> rectify/SGBM",
             "propagates_to_geometry_frame": "indirectly"},
            {"name": "left_timestamp", "type": "Optional[float]", "required": False,
             "semantics": "opaque caller-defined float; no unit enforced",
             "ownership": "value", "read_by_dpe": True,
             "consumed_at": "pipeline.py result_timestamp (takes precedence over right_timestamp)",
             "propagates_to_geometry_frame": "YES -> GeometryFrame.timestamp"},
            {"name": "right_timestamp", "type": "Optional[float]", "required": False,
             "semantics": "fallback timestamp, used ONLY when left_timestamp is None",
             "ownership": "value", "read_by_dpe": True,
             "consumed_at": "pipeline.py result_timestamp fallback",
             "propagates_to_geometry_frame": "only when left_timestamp is None"},
            {"name": "calibration", "type": "Optional[StereoCalibration]", "required": False,
             "semantics": "RESERVED for future multi-rig use", "ownership": "reference",
             "read_by_dpe": False,
             "consumed_at": "NOWHERE — pipeline always uses its constructor calibration",
             "propagates_to_geometry_frame": "NO"},
            {"name": "frame_id", "type": "Optional[str]", "required": False,
             "semantics": "declared as observation identity in the class docstring",
             "ownership": "value", "read_by_dpe": False,
             "consumed_at": "NOWHERE — never destructured in process_observation()",
             "propagates_to_geometry_frame": "NO"},
            {"name": "motion_hint", "type": "Optional[MotionHint]", "required": False,
             "semantics": "single motion sample associated with this capture",
             "ownership": "reference", "read_by_dpe": True,
             "consumed_at": "TemporalRecord.motion_hint (association only, contents never read)",
             "propagates_to_geometry_frame": "NO (stored in internal TemporalHistory only)"},
            {"name": "motion_hints", "type": "Optional[Sequence[MotionHint]]", "required": False,
             "semantics": "bounded window of motion samples over the inter-frame interval",
             "ownership": "reference", "read_by_dpe": True,
             "consumed_at": "compute_rotation_compensation + compute_motion_aware_reliability + persistence",
             "propagates_to_geometry_frame": "indirectly via rotation_compensation_status / motion_aware_reliability"},
        ],
        "probe_observation_frame_id": "observation-X",
        "geometry_frame_frame_id": gf.frame_id,
        "geometry_frame_frame_id_is_coordinate_frame": gf.frame_id == FrameId.CAMERA_OPTICAL_LEFT,
        "observation_id_found_anywhere_in_geometry_frame": scan_for("observation-X", gf),
        "observation_id_found_in_legacy_result": legacy_fields_with_token,
        "production_source_reads_of_observation_frame_id": reads,
        "nested_frame_id_values": nested_frame_ids,
        "geometry_frame_identity_fields": ["timestamp", "frame_id"],
        "exact_join_key_available": False,
    }


# =====================================================================
# 5. Timestamp semantics
# =====================================================================
def section_timestamps() -> Dict[str, Any]:
    cal = F.calibration()
    left, right = F.scene_pair()
    out: Dict[str, Any] = {}

    # source / precedence
    p = DepthPerceptionPipeline(F.qualified_config(), cal, rectify=True, body_T_camera_left=F.body_transform())
    gf_left_only = p.process_geometry_frame(
        StereoObservation(left_image=left, right_image=right, left_timestamp=10.0, right_timestamp=99.0)
    )
    p.reset()
    gf_right_only = p.process_geometry_frame(
        StereoObservation(left_image=left, right_image=right, left_timestamp=None, right_timestamp=42.0)
    )
    p.reset()
    gf_none = p.process_geometry_frame(
        StereoObservation(left_image=left, right_image=right)
    )
    out["left_wins_when_both_present"] = {"left": 10.0, "right": 99.0, "geometry_frame_timestamp": gf_left_only.timestamp}
    out["right_used_when_left_none"] = {"right": 42.0, "geometry_frame_timestamp": gf_right_only.timestamp}
    out["none_when_neither_supplied"] = {"geometry_frame_timestamp": gf_none.timestamp}
    out["accepted_type"] = "float (opaque); no unit conversion, no unit validation anywhere in DPE"
    out["units_defined"] = False
    out["units_note"] = (
        "Docstrings call the value 'opaque caller-defined'. Every temporal threshold "
        "(temporal_max_age_s / temporal_gap_limit_s) is named *_s and compared directly "
        "against timestamp differences, so SECONDS is the de facto required unit whenever "
        "enable_temporal is True — but nothing validates or enforces it."
    )

    # monotonicity / duplicates / decreasing, observed through the public contract
    seq = []
    p2 = DepthPerceptionPipeline(F.qualified_config(), cal, rectify=True, body_T_camera_left=F.body_transform())
    for ts in [1.0, 2.0, 2.0, 1.5, 3.0, float("nan"), None, 4.0]:
        r = p2.process_observation(StereoObservation(left_image=left, right_image=right, left_timestamp=ts))
        seq.append({
            "submitted_timestamp": jsonable(ts),
            "admission_status": r.temporal_admission_status,
            "result_timestamp": jsonable(r.timestamp),
            "history_len": len(p2.temporal_history),
            "temporal_consistency_state": r.temporal_consistency.state if r.temporal_consistency else None,
        })
    out["chronology_probe"] = seq
    out["monotonicity_enforced_for_temporal_history"] = True
    out["monotonicity_enforced_for_geometry_output"] = False
    out["duplicate_timestamp_accepted_by_history"] = False
    out["decreasing_timestamp_accepted_by_history"] = False
    out["geometry_still_produced_when_history_rejects"] = True
    out["timestamp_sufficient_as_exact_join_key"] = False
    return out


# =====================================================================
# 6. GeometryFrame contract enumeration
# =====================================================================
_GF_GROUPS = {
    "timestamp": ("identity/provenance", "optional (None when caller supplies no timestamp)"),
    "frame_id": ("identity/provenance", "mandatory (always the COORDINATE frame camera_optical_left)"),
    "disparity_map": ("raster geometry", "mandatory"),
    "depth_map": ("raster geometry", "mandatory"),
    "valid_disparity_mask": ("raster geometry", "mandatory in practice (always built by build_result)"),
    "valid_depth_mask": ("raster geometry", "mandatory in practice"),
    "geometry": ("3-D geometry", "optional/gated on enable_geometry"),
    "geometry_body": ("3-D geometry", "optional/gated on enable_geometry AND body_T_camera_left"),
    "obstacle_cloud": ("obstacle/free-space evidence", "optional/gated on geometry_body AND enable_obstacle_geometry"),
    "free_space_rays": ("obstacle/free-space evidence", "optional/gated on geometry_body AND enable_free_space_rays"),
    "geometry_metrics": ("quality", "optional/gated on geometry_body"),
    "temporal_consistency": ("temporal evidence", "temporally derived; None when not admitted/no prior"),
    "temporal_stabilization": ("temporal evidence", "temporally derived; gated on enable_temporal_stabilization"),
    "rotation_compensation_status": ("temporal evidence", "temporally derived; gated on enable_rotation_compensation"),
    "motion_aware_reliability": ("temporal evidence", "temporally derived; gated on enable_motion_aware_reliability"),
    "temporal_persistence": ("temporal evidence", "temporally derived; gated on all three temporal flags"),
    "region_evidence": ("legacy-derived", "always populated (extracted from traversability layer)"),
    "clearance_evidence": ("clearance evidence", "always populated (extracted from ThreatAssessor beams)"),
    "surface_evidence": ("surface evidence", "optional/gated on enable_surface_geometry AND geometry_body"),
    "boundary_evidence": ("boundary evidence", "optional/gated on enable_boundary_geometry only"),
    "opening_evidence": ("opening evidence", "optional/gated; POSITIVE-FINDINGS-ONLY list"),
    "quality": ("quality", "always populated when the frame is built"),
}


def _describe_field(name: str, value: Any) -> Dict[str, Any]:
    group, gating = _GF_GROUPS.get(name, ("metadata", "unknown"))
    d: Dict[str, Any] = {
        "field": name,
        "group": group,
        "classification": gating,
        "present_under_qualified_config": value is not None,
        "python_type": type(value).__name__,
    }
    if isinstance(value, np.ndarray):
        d["shape"] = list(value.shape)
        d["dtype"] = str(value.dtype)
    elif isinstance(value, (list, tuple)):
        d["len"] = len(value)
        d["empty"] = len(value) == 0
    elif isinstance(value, dict):
        d["len"] = len(value)
        d["keys"] = sorted(value.keys())
    elif hasattr(value, "state"):
        d["state"] = getattr(value, "state")
    elif isinstance(value, (str, float, int)) or value is None:
        d["value"] = jsonable(value)
    return d


def section_geometry_frame_contract() -> Dict[str, Any]:
    cal = F.calibration()
    left, right = F.two_object_scene()

    def build(cfg, transform, n=3):
        p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=transform)
        gf = None
        for i in range(n):
            gf = p.process_geometry_frame(
                observation(left, right, ts=float(i) * 0.1,
                            hints=F.motion_hint_window((i - 1) * 0.1, i * 0.1) if i else None)
            )
        return gf

    full = build(F.qualified_config(), F.body_transform())
    minimal = build(dpe.PipelineConfig(enable_geometry_frame=True), None, n=1)

    fields = list(GeometryFrame.__slots__)
    return {
        "field_count": len(fields),
        "fields_under_qualified_config": [_describe_field(f, getattr(full, f)) for f in fields],
        "fields_under_default_config": [
            {"field": f, "present": getattr(minimal, f) is not None} for f in fields
        ],
        "quality_under_qualified_config": {
            "overall_state": full.quality.overall_state,
            "geometry_validity_state": full.quality.geometry_validity_state,
            "temporal_consistency_state": full.quality.temporal_consistency_state,
            "motion_reliability_state": full.quality.motion_reliability_state,
            "persistence_state": full.quality.persistence_state,
            "degradation_reasons": list(full.quality.degradation_reasons),
        },
    }


# =====================================================================
# 7. Mutable state inventory
# =====================================================================
def section_mutable_state() -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=F.body_transform())
    left, right = F.scene_pair()

    def snapshot():
        return {
            name: {
                "type": type(getattr(p, name)).__name__,
                "id": id(getattr(p, name)),
                "repr": repr(getattr(p, name))[:120],
            }
            for name in DepthPerceptionPipeline.__dict__.get("__slots__", None) or [
                a for a in vars(p)
            ]
        }

    before = snapshot()
    for i in range(5):
        p.process_geometry_frame(
            observation(left, right, ts=float(i) * 0.1,
                        hints=F.motion_hint_window((i - 1) * 0.1, i * 0.1) if i else None)
        )
    after = snapshot()

    changed_identity = sorted(k for k in before if before[k]["id"] != after[k]["id"])
    changed_repr = sorted(k for k in before if before[k]["repr"] != after[k]["repr"])

    # Internal collaborator state
    ta = p._threat_assessor
    th = p._temporal_history
    tp = p._temporal_persistence_tracker

    classification = {
        "_config": ("A", "PipelineConfig", "construction", "immutable configuration; never mutated per frame"),
        "_calibration": ("A", "StereoCalibration", "construction", "frozen calibration value object"),
        "_rectify": ("A", "bool", "construction", "flag"),
        "_body_T_camera_left": ("A", "RigidTransform", "construction", "frozen extrinsic"),
        "_rectifier": ("A", "RectificationEngine", "construction",
                       "holds precomputed cv2 rectification maps; read-only per frame"),
        "_disparity_engine": ("A", "DisparityEngine", "construction",
                              "wraps a cv2.StereoSGBM object; compute() writes no Python-visible state, "
                              "but the underlying OpenCV matcher owns internal scratch buffers"),
        "_depth_estimator": ("A", "DepthEstimator", "construction", "holds Q; pure function per frame"),
        "_point_cloud_builder": ("A", "PointCloudBuilder", "construction", "holds Q; allocates fresh arrays per call"),
        "_scene_interpreter": ("A", "SceneInterpreter", "construction", "stateless analyzer (grid params only)"),
        "_rectified_focal_length_px": ("A", "float", "construction", "derived scalar"),
        "_rectified_principal_point_px": ("A", "tuple", "construction", "derived scalar pair"),
        "_threat_assessor": ("B", "ThreatAssessor", "per-instance, rebuilt by reset()",
                             "per-beam EMA + debounce state — frame-to-frame semantics depend on it"),
        "_temporal_history": ("B", "TemporalHistory", "per-instance, cleared by reset()",
                              "bounded chronology of TemporalRecord; drives consistency/stabilization"),
        "_temporal_persistence_tracker": ("B", "TemporalPersistenceTracker", "per-instance, cleared by reset()",
                                          "per-cell support/absence counters"),
        "_closed": ("C", "bool", "per-instance", "lifecycle flag"),
        "_frames_processed": ("C", "int", "per-instance, zeroed by reset()", "counter"),
        "_last_confidence": ("C", "Optional[float]", "per-instance, cleared by reset()", "bookkeeping"),
        "_last_processing_time_ms": ("C", "Optional[float]", "per-instance, cleared by reset()", "bookkeeping"),
    }

    inventory = []
    for name, (cls, typ, lifetime, purpose) in classification.items():
        present = hasattr(p, name)
        inventory.append({
            "attribute": name,
            "class": cls,
            "type": typ,
            "lifetime": lifetime,
            "purpose": purpose,
            "present": present,
            "value_is_none": getattr(p, name, "MISSING") is None,
            "rebound_during_5_frames": name in changed_identity,
            "observably_changed_during_5_frames": name in changed_repr,
        })

    return {
        "instance_attributes": sorted(vars(p).keys()),
        "attributes_rebound_across_5_frames": changed_identity,
        "attributes_whose_repr_changed": changed_repr,
        "inventory": inventory,
        "threat_assessor_internal_state": sorted(vars(ta).keys()),
        "temporal_history_len_after_5": len(th) if th is not None else None,
        "temporal_history_max_records": cfg.temporal_max_records,
        "persistence_tracker_internal_state": sorted(vars(tp).keys()) if tp is not None else None,
        "frames_processed": p.health().frames_processed,
    }


# =====================================================================
# 8. Reentrancy evidence (NO threads are created anywhere)
# =====================================================================
def section_reentrancy_evidence() -> Dict[str, Any]:
    """Evidence-gathering only. No thread, process, future or executor is
    created here — the question 'would two simultaneous calls be safe?' is
    answered from observed per-frame mutation of shared instance state."""
    cal = F.calibration()
    p = DepthPerceptionPipeline(F.qualified_config(), cal, rectify=True, body_T_camera_left=F.body_transform())
    left, right = F.two_object_scene()

    trace = []
    for i in range(4):
        ta = p._threat_assessor
        tp = p._temporal_persistence_tracker
        th = p._temporal_history
        before = {
            "ema_dist_sum": float(np.nansum(ta._ema_dist)) if ta._ema_dist is not None else None,
            "pending_count_sum": int(np.sum(ta._pending_count)) if ta._pending_count is not None else None,
            "history_len": len(th),
            "persistence_support_sum": (
                int(tp._support_count.sum()) if tp is not None and tp._support_count is not None else None
            ),
        }
        p.process_geometry_frame(
            observation(left, right, ts=float(i) * 0.1,
                        hints=F.motion_hint_window((i - 1) * 0.1, i * 0.1) if i else None)
        )
        after = {
            "ema_dist_sum": float(np.nansum(ta._ema_dist)) if ta._ema_dist is not None else None,
            "pending_count_sum": int(np.sum(ta._pending_count)) if ta._pending_count is not None else None,
            "history_len": len(th),
            "persistence_support_sum": (
                int(tp._support_count.sum()) if tp is not None and tp._support_count is not None else None
            ),
        }
        trace.append({"frame": i, "before": before, "after": after,
                      "mutated": before != after})

    # Structural evidence: is there ANY lock / synchronization primitive in
    # production source?
    src_root = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")
    sync_hits, thread_hits = [], []
    for dirpath, _d, files in os.walk(src_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    s = line.strip()
                    if s.startswith("#"):
                        continue
                    if re.search(r"\b(Lock|RLock|Semaphore|Condition|threading|asyncio|Queue|Executor)\b", s):
                        sync_hits.append(f"{os.path.relpath(path, _REPO_ROOT)}:{i}: {s[:110]}")
                    if re.search(r"^\s*import\s+(threading|queue|asyncio|multiprocessing|concurrent)", line):
                        thread_hits.append(f"{os.path.relpath(path, _REPO_ROOT)}:{i}: {s[:110]}")

    return {
        "per_frame_shared_state_mutation": trace,
        "all_frames_mutated_shared_state": all(t["mutated"] for t in trace),
        "synchronization_primitives_in_production_source": sync_hits,
        "concurrency_imports_in_production_source": thread_hits,
        "cv2_stereo_sgbm_instance_shared_across_calls": True,
        "classification": "NOT THREAD-SAFE / ONE ACTIVE CALL AT A TIME",
        "threads_created_by_this_audit": 0,
    }


# =====================================================================
# 9. Construction cost + resource reuse
# =====================================================================
def section_construction(n: int = 40) -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    left, right = F.scene_pair()

    samples = []
    for _ in range(5):  # warm up
        DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    for _ in range(n):
        t0 = time.perf_counter()
        DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples_norect = []
    for _ in range(n):
        t0 = time.perf_counter()
        DepthPerceptionPipeline(cfg, cal, rectify=False, body_T_camera_left=tf)
        samples_norect.append((time.perf_counter() - t0) * 1000.0)

    # Proof of reuse: object identities across 20 frames
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    watched = [
        "_rectifier", "_disparity_engine", "_depth_estimator", "_point_cloud_builder",
        "_scene_interpreter", "_threat_assessor", "_temporal_history",
        "_temporal_persistence_tracker",
    ]
    ids0 = {w: id(getattr(p, w)) for w in watched}
    sgbm0 = id(getattr(p._disparity_engine, "_matcher", None) or
               getattr(p._disparity_engine, "_stereo", None) or p._disparity_engine)
    maps0 = [id(m) for m in (
        getattr(p._rectifier, "_map1x", None), getattr(p._rectifier, "_map1y", None),
        getattr(p._rectifier, "_map2x", None), getattr(p._rectifier, "_map2y", None),
    )]
    for i in range(20):
        p.process_geometry_frame(observation(left, right, ts=float(i) * 0.1))
    ids1 = {w: id(getattr(p, w)) for w in watched}
    sgbm1 = id(getattr(p._disparity_engine, "_matcher", None) or
               getattr(p._disparity_engine, "_stereo", None) or p._disparity_engine)
    maps1 = [id(m) for m in (
        getattr(p._rectifier, "_map1x", None), getattr(p._rectifier, "_map1y", None),
        getattr(p._rectifier, "_map2x", None), getattr(p._rectifier, "_map2y", None),
    )]

    return {
        "construction_rectify_true_ms": stats_ms(samples),
        "construction_rectify_false_ms": stats_ms(samples_norect),
        "rectification_map_build_cost_ms_est": (
            float(np.median(samples)) - float(np.median(samples_norect))
        ),
        "disparity_engine_attrs": sorted(vars(p._disparity_engine).keys()),
        "rectifier_attrs": sorted(vars(p._rectifier).keys()),
        "object_identity_stable_over_20_frames": {w: ids0[w] == ids1[w] for w in watched},
        "sgbm_matcher_identity_stable": sgbm0 == sgbm1,
        "rectification_map_identity_stable": maps0 == maps1,
        "constructed_once_reused_per_frame": True,
    }


# =====================================================================
# 10. Reset / close semantics
# =====================================================================
def section_reset_close() -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=F.body_transform())
    left, right = F.two_object_scene()

    for i in range(5):
        p.process_geometry_frame(
            observation(left, right, ts=float(i) * 0.1,
                        hints=F.motion_hint_window((i - 1) * 0.1, i * 0.1) if i else None)
        )

    def snap():
        ta, th, tp = p._threat_assessor, p._temporal_history, p._temporal_persistence_tracker
        return {
            "threat_assessor_id": id(ta),
            "ema_dist_sum": float(np.nansum(ta._ema_dist)) if ta._ema_dist is not None else None,
            "ema_dist_nonzero_count": int(np.count_nonzero(np.nan_to_num(ta._ema_dist))) if ta._ema_dist is not None else None,
            "pending_count_sum": int(np.sum(ta._pending_count)) if ta._pending_count is not None else None,
            "history_len": len(th) if th is not None else None,
            "persistence_support_sum": (
                int(tp._support_count.sum()) if tp is not None and tp._support_count is not None else None
            ),
            "persistence_grid_shape": (
                jsonable(tp._grid_shape) if tp is not None else None
            ),
            "frames_processed": p.health().frames_processed,
            "last_confidence": jsonable(p.health().last_confidence),
            "last_processing_time_ms": p.health().last_processing_time_ms is not None,
            "rectifier_id": id(p._rectifier),
            "disparity_engine_id": id(p._disparity_engine),
            "calibration_id": id(p.calibration),
            "config_id": id(p.config),
        }

    before = snap()
    p.reset()
    after = snap()

    cleared = sorted(k for k in before if before[k] != after[k])
    retained = sorted(k for k in before if before[k] == after[k])

    # close() behaviour
    p2 = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=F.body_transform())
    p2.process_geometry_frame(observation(left, right, ts=0.0))
    p2.close()
    close_effects = {}
    for label, fn in (
        ("process_geometry_frame", lambda: p2.process_geometry_frame(observation(left, right, ts=1.0))),
        ("process_observation", lambda: p2.process_observation(observation(left, right, ts=1.0))),
        ("process", lambda: p2.process(left, right)),
        ("reset", lambda: p2.reset()),
        ("health", lambda: p2.health()),
        ("close_again", lambda: p2.close()),
    ):
        try:
            fn()
            close_effects[label] = "OK (no exception)"
        except Exception as exc:
            close_effects[label] = f"{type(exc).__name__}: {exc}"

    return {
        "reset_changed": cleared,
        "reset_retained": retained,
        "before_reset": before,
        "after_reset": after,
        "threat_assessor_rebuilt_not_mutated": before["threat_assessor_id"] != after["threat_assessor_id"],
        "close_effects": close_effects,
        "close_is_idempotent": close_effects.get("close_again") == "OK (no exception)",
        "health_still_readable_after_close": close_effects.get("health", "").startswith("OK"),
    }


# =====================================================================
# 12. Calibration + physical unit contract
# =====================================================================
def section_unit_contract() -> Dict[str, Any]:
    from depth_perception_engine.depth.depth_estimator import DepthEstimator

    cal = F.calibration()
    Q = cal.Q
    f_px = abs(float(Q[2, 3]))
    q32 = float(Q[3, 2])
    baseline_q_units = abs(1.0 / q32)
    est = DepthEstimator.from_calibration(cal)

    # Closed-form round trip: disparity for a plane at Z metres, then read
    # DPE's own depth back, using DPE's own DepthEstimator.
    checks = []
    for z_m in (0.5, 1.0, 2.0, 4.0, 8.0):
        # In DPE's convention Q's translation is in mm, so
        # disparity = f_px * baseline_mm / Z_mm = f_px * baseline_mm / (Z_m * 1000)
        d = f_px * baseline_q_units / (z_m * 1000.0)
        disp = np.full((8, 8), d, dtype=np.float32)
        depth = est.estimate(disp)
        checks.append({
            "target_depth_m": z_m,
            "disparity_px": float(d),
            "dpe_depth_m": float(np.median(depth)),
            "abs_error_m": float(abs(np.median(depth) - z_m)),
        })

    # What if a producer instead wrote the baseline in METRES (0.12) while
    # DPE expects mm? Build the two candidate Q matrices for a 0.12 m rig
    # and read DPE's own depth back for a plane at a known distance.
    def q_with_baseline(value_in_q_units: float) -> np.ndarray:
        q = Q.astype(np.float64).copy()
        q[3, 2] = 1.0 / value_in_q_units
        return q

    scenarios = {}
    true_z_m = 2.0
    for label, baseline_value, intended_m in (
        ("gazebo_writes_120.0_mm_convention", 120.0, 0.12),
        ("gazebo_writes_0.12_metre_convention", 0.12, 0.12),
    ):
        q = q_with_baseline(baseline_value)
        e = DepthEstimator(q)
        # true disparity produced by a physically 0.12 m rig at true_z_m,
        # with the same rectified focal length
        d_true = f_px * (intended_m * 1000.0) / (true_z_m * 1000.0)
        disp = np.full((8, 8), d_true, dtype=np.float32)
        depth = e.estimate(disp)
        med = float(np.median(depth))
        scenarios[label] = {
            "Q[3,2]": float(q[3, 2]),
            "derived_baseline_in_q_units": float(abs(1.0 / q[3, 2])),
            "true_scene_depth_m": true_z_m,
            "true_disparity_px": float(d_true),
            "dpe_reported_depth_m": med if med != 0.0 else 0.0,
            "dpe_reported_depth_all_invalid": bool(np.all(depth == 0.0)),
            "correct": bool(abs(med - true_z_m) < 0.01),
        }

    return {
        "authoritative_unit_contract": {
            "calibration_Q_translation_units": "MILLIMETRES",
            "evidence_source": "src/depth_perception_engine/depth/depth_estimator.py:132-133 "
                               "('Calibration object-points were in mm -> Z comes out in mm; convert to m')",
            "focal_length_units": "pixels (Q[2,3])",
            "principal_point_units": "pixels (-Q[0,3], -Q[1,3])",
            "disparity_units": "pixels; invalid convention is disparity <= 0",
            "depth_map_units": "METRES (float32); invalid is exactly 0.0",
            "point_cloud_units": "METRES; invalid is NaN",
            "obstacle_distance_units": "METRES",
            "free_space_range_units": "METRES",
            "clearance_distance_units": "METRES; bearings in RADIANS",
            "body_transform_translation_units": "METRES (frames.RigidTransform)",
            "depth_clamp_m": [DepthEstimator.MIN_DEPTH_M, DepthEstimator.MAX_DEPTH_M],
        },
        "fixture_calibration": {
            "image_size": list(cal.image_size),
            "rectified_focal_length_px": f_px,
            "Q[3,2]": q32,
            "baseline_in_Q_units": baseline_q_units,
            "baseline_interpreted_as_mm": baseline_q_units,
            "baseline_interpreted_as_m": baseline_q_units / 1000.0,
            "documented_rig_baseline": "64 mm (config/pipeline_config.py and depth_estimator.py comments)",
            "conclusion": "Q translation is expressed in MILLIMETRES.",
        },
        "closed_form_round_trip": checks,
        "gazebo_baseline_120_scenarios": scenarios,
    }


# =====================================================================
# 11. Motion / IMU path
# =====================================================================
def section_motion_path(n: int = 60) -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(n + 10)

    def run(with_motion: bool, malformed: Optional[str] = None):
        p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
        samples, statuses, rel_states = [], [], []
        for i, (l, r) in enumerate(frames):
            ts = float(i) * 0.1
            hints = None
            if with_motion and i > 0:
                hints = F.motion_hint_window(ts - 0.1, ts, wz=0.05)
                if malformed == "nan":
                    hints = [F.motion_hint(ts - 0.1, wz=float("nan")), F.motion_hint(ts, wz=float("nan"))]
                elif malformed == "invalid_flag":
                    hints = [F.motion_hint(ts - 0.1, wz=0.05, valid=False), F.motion_hint(ts, wz=0.05, valid=False)]
                elif malformed == "empty":
                    hints = []
                elif malformed == "stale":
                    hints = F.motion_hint_window(ts - 100.0, ts - 99.0, wz=0.05)
            t0 = time.perf_counter()
            gf = p.process_geometry_frame(observation(l, r, ts=ts, hints=hints))
            dt = (time.perf_counter() - t0) * 1000.0
            if i >= 10:
                samples.append(dt)
            statuses.append(gf.rotation_compensation_status)
            rel_states.append(gf.motion_aware_reliability.state if gf.motion_aware_reliability else None)
        return samples, statuses, rel_states, gf

    no_m_samples, no_m_status, no_m_rel, gf_none = run(False)
    m_samples, m_status, m_rel, gf_motion = run(True)

    malformed_results = {}
    for label in ("nan", "invalid_flag", "empty", "stale"):
        try:
            s, st, rel, gf = run(True, malformed=label)
            malformed_results[label] = {
                "raised": False,
                "median_ms": float(np.median(s)),
                "rotation_compensation_status_final": st[-1],
                "motion_reliability_final": rel[-1],
                "unique_rotation_statuses": sorted({str(x) for x in st}),
                "unique_reliability_states": sorted({str(x) for x in rel}),
            }
        except Exception as exc:
            malformed_results[label] = {"raised": True, "error": f"{type(exc).__name__}: {exc}"}

    def _diff(a: GeometryFrame, b: GeometryFrame) -> Dict[str, Any]:
        return {
            "depth_map_identical": bool(np.array_equal(a.depth_map, b.depth_map)),
            "disparity_map_identical": bool(np.array_equal(a.disparity_map, b.disparity_map)),
            "rotation_compensation_status": [a.rotation_compensation_status, b.rotation_compensation_status],
            "motion_aware_reliability_state": [
                a.motion_aware_reliability.state if a.motion_aware_reliability else None,
                b.motion_aware_reliability.state if b.motion_aware_reliability else None,
            ],
            "temporal_consistency_state": [
                a.temporal_consistency.state if a.temporal_consistency else None,
                b.temporal_consistency.state if b.temporal_consistency else None,
            ],
            "temporal_persistence_state": [
                a.temporal_persistence.state if a.temporal_persistence else None,
                b.temporal_persistence.state if b.temporal_persistence else None,
            ],
            "quality_overall_state": [a.quality.overall_state, b.quality.overall_state],
            "quality_degradation_reasons": [
                list(a.quality.degradation_reasons), list(b.quality.degradation_reasons)
            ],
        }

    return {
        "authoritative_motion_field": "motion_hints (Sequence[MotionHint])",
        "motion_hint_singular_usage": (
            "attached unread to TemporalRecord.motion_hint; its angular_velocity_rad_s is "
            "never integrated by any DPE algorithm"
        ),
        "angular_velocity_units": "radians per second, (3,) ndarray about frame_id's X/Y/Z",
        "motion_frame": "declared per-sample via MotionHint.frame_id; expected FrameId.BODY, never inferred",
        "motion_timestamp_units": "same opaque float convention as StereoObservation timestamps",
        "no_motion_ms": stats_ms(no_m_samples),
        "with_motion_ms": stats_ms(m_samples),
        "motion_overhead_median_ms": float(np.median(m_samples) - np.median(no_m_samples)),
        "no_motion_rotation_statuses": sorted({str(x) for x in no_m_status}),
        "with_motion_rotation_statuses": sorted({str(x) for x in m_status}),
        "no_motion_reliability_states": sorted({str(x) for x in no_m_rel}),
        "with_motion_reliability_states": sorted({str(x) for x in m_rel}),
        "output_difference_last_frame": _diff(gf_none, gf_motion),
        "malformed_motion": malformed_results,
    }


# =====================================================================
# 18. Frame sequence / drop / discontinuity tolerance
# =====================================================================
def section_discontinuity() -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(12)

    def run(timestamps, label, hints_fn=None):
        p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
        rows = []
        for i, ts in enumerate(timestamps):
            l, r = frames[i % len(frames)]
            hints = hints_fn(i, ts) if hints_fn else None
            gf = p.process_geometry_frame(observation(l, r, ts=ts, hints=hints))
            res_ts = gf.timestamp
            rows.append({
                "i": i,
                "submitted_timestamp": jsonable(ts),
                "geometry_frame_timestamp": jsonable(res_ts),
                "history_len": len(p.temporal_history),
                "temporal_consistency": gf.temporal_consistency.state if gf.temporal_consistency else None,
                "temporal_stabilization": gf.temporal_stabilization.state if gf.temporal_stabilization else None,
                "rotation_compensation": gf.rotation_compensation_status,
                "motion_reliability": gf.motion_aware_reliability.state if gf.motion_aware_reliability else None,
                "persistence": gf.temporal_persistence.state if gf.temporal_persistence else None,
                "quality_overall": gf.quality.overall_state,
                "geometry_present": gf.geometry is not None,
                "depth_valid_fraction": float(np.mean(gf.depth_map > 0.0)),
            })
        return {"label": label, "rows": rows}

    dt = 0.1
    contiguous = run([i * dt for i in range(10)], "contiguous_100..109")
    # observation ids 100, 101, 105 -> intermediate captures never submitted
    skipped = run([0.0, dt, 5 * dt, 6 * dt, 10 * dt], "skipped_100_101_105_106_110")
    duplicate = run([0.0, dt, dt, 2 * dt], "duplicate_timestamp")
    decreasing = run([0.0, dt, 0.5 * dt, 2 * dt], "decreasing_timestamp")
    # gap_limit_s is 5.0 in the qualified config
    small_gap = run([0.0, dt, dt + 1.0, dt + 1.1], "forward_jump_1.0s_within_gap_limit")
    big_jump = run([0.0, dt, dt + 30.0, dt + 30.1], "forward_jump_30s_beyond_gap_limit")

    return {
        "temporal_history_keys_on": "TemporalRecord.timestamp ONLY",
        "evidence": (
            "temporal/history.py::TemporalHistory.admit() compares only record.timestamp against "
            "self._records[-1].timestamp; there is no sequence number, no arrival counter, and no "
            "wall-clock read (time.time/time.perf_counter is never called in that module)."
        ),
        "sequence_number_exists_anywhere": False,
        "arrival_order_matters": True,
        "arrival_order_note": (
            "Arrival order matters only in that admit() compares against the NEWEST ADMITTED record, "
            "so an out-of-order submission is rejected rather than re-sorted. History is never reordered."
        ),
        "persistence_counts_processed_frames_not_captured_frames": True,
        "scenarios": {
            s["label"]: s["rows"] for s in
            (contiguous, skipped, duplicate, decreasing, small_gap, big_jump)
        },
        "classification": "A — algorithm remains valid; temporal confidence/history reflect the larger gap",
        "hpe_may_drop_before_dpe": True,
    }


# =====================================================================
# 19. Failure semantics
# =====================================================================
def section_failure_semantics() -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    left, right = F.scene_pair()
    h, w = left.shape[:2]

    def probe(label, make_obs, classification, fresh=True, pipeline=None):
        p = pipeline or DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
        entry: Dict[str, Any] = {"case": label, "classification": classification}
        try:
            gf = p.process_geometry_frame(make_obs())
            entry.update({
                "outcome": "returned GeometryFrame",
                "raised": None,
                "quality_overall": gf.quality.overall_state,
                "degradation_reasons": list(gf.quality.degradation_reasons),
                "geometry_validity_state": gf.quality.geometry_validity_state,
                "depth_valid_fraction": float(np.mean(gf.depth_map > 0.0)),
                "geometry_present": gf.geometry is not None,
                "obstacle_cloud_points": (
                    int(gf.obstacle_cloud.points.shape[0]) if gf.obstacle_cloud is not None else None
                ),
                "opening_evidence_count": len(gf.opening_evidence) if gf.opening_evidence is not None else None,
            })
        except Exception as exc:
            entry.update({
                "outcome": "RAISED",
                "raised": f"{type(exc).__name__}",
                "message": str(exc)[:220],
            })
        return entry

    cases = []
    cases.append(probe("well_formed_baseline", lambda: observation(left, right, ts=0.0),
                       "n/a — control"))
    cases.append(probe("mismatched_dimensions",
                       lambda: observation(left, cv2.resize(right, (w // 2, h // 2)), ts=0.0),
                       "CALLER CONTRACT VIOLATION"))
    cases.append(probe("wrong_frame_size_vs_calibration",
                       lambda: observation(cv2.resize(left, (w // 2, h // 2)),
                                           cv2.resize(right, (w // 2, h // 2)), ts=0.0),
                       "CALLER CONTRACT VIOLATION"))
    cases.append(probe("left_is_none", lambda: observation(None, right, ts=0.0),
                       "CALLER CONTRACT VIOLATION"))
    cases.append(probe("float64_dtype",
                       lambda: observation(left.astype(np.float64), right.astype(np.float64), ts=0.0),
                       "CALLER CONTRACT VIOLATION"))
    cases.append(probe("grayscale_pair",
                       lambda: observation(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                                           cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), ts=0.0),
                       "n/a — supported input"))
    cases.append(probe("channel_count_mismatch",
                       lambda: observation(left, cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), ts=0.0),
                       "CALLER CONTRACT VIOLATION"))
    cases.append(probe("textureless_no_disparity",
                       lambda: observation(np.full_like(left, 128), np.full_like(right, 128), ts=0.0),
                       "RECOVERABLE FRAME DEGRADATION"))
    cases.append(probe("decorrelated_pair_no_valid_depth",
                       lambda: (lambda lr: observation(lr[0], lr[1], ts=0.0))(
                           __import__("benchmarks.i6_temporal.fixtures", fromlist=["decorrelated_pair"]
                                      ).decorrelated_pair(7)),
                       "RECOVERABLE FRAME DEGRADATION"))
    cases.append(probe("missing_motion", lambda: observation(left, right, ts=0.0, hints=None),
                       "n/a — legal and unremarkable"))
    cases.append(probe("malformed_motion_nan",
                       lambda: observation(left, right, ts=0.0,
                                           hints=[F.motion_hint(-0.1, wz=float("nan")),
                                                  F.motion_hint(0.0, wz=float("nan"))]),
                       "TEMPORAL DEGRADATION"))
    cases.append(probe("no_timestamp_temporal_rejected",
                       lambda: observation(left, right, ts=None),
                       "TEMPORAL DEGRADATION"))
    cases.append(probe("nan_timestamp",
                       lambda: observation(left, right, ts=float("nan")),
                       "TEMPORAL DEGRADATION"))

    # temporal admission rejection on a live pipeline
    p_seq = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    p_seq.process_geometry_frame(observation(left, right, ts=5.0))
    cases.append(probe("temporal_admission_rejected_older_timestamp",
                       lambda: observation(left, right, ts=1.0),
                       "TEMPORAL DEGRADATION", pipeline=p_seq))

    # after close()
    p_closed = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    p_closed.close()
    cases.append(probe("call_after_close", lambda: observation(left, right, ts=0.0),
                       "FATAL ENGINE ERROR", pipeline=p_closed))

    # calibration mismatch: pipeline built for one size, image built for the derived 2x size
    big_cal = F.scaled_calibration(cal, 2)
    big_l, big_r = F.upscale_pair(left, right, 2)
    cases.append(probe("image_matches_a_different_calibration",
                       lambda: observation(big_l, big_r, ts=0.0),
                       "CALLER CONTRACT VIOLATION"))

    # rectify=False escape hatch with mismatched size (no rectification validation)
    p_norect = DepthPerceptionPipeline(cfg, cal, rectify=False, body_T_camera_left=tf)
    cases.append(probe("rectify_false_with_oversized_image",
                       lambda: observation(big_l, big_r, ts=0.0),
                       "CALLER CONTRACT VIOLATION (silent)", pipeline=p_norect))

    return {
        "cases": cases,
        "note": (
            "No DPE stage is wrapped in try/except: pipeline.py deliberately lets rectification, "
            "point-cloud and geometry failures propagate so a bad frame is dropped rather than "
            "silently trusted (see pipeline.py's comments at the rectify and geometry stages)."
        ),
    }


# =====================================================================
# 13. Authoritative synchronous benchmark
# =====================================================================
def _bench_geometry_frame(cfg, cal, tf, frames, rectify, n_warmup, n_iters, with_motion=True, dt=0.1):
    p = DepthPerceptionPipeline(cfg, cal, rectify=rectify, body_T_camera_left=tf)
    samples: List[float] = []
    total = n_warmup + n_iters
    last_gf = None
    for i in range(total):
        l, r = frames[i % len(frames)]
        ts = float(i) * dt
        hints = F.motion_hint_window(ts - dt, ts, wz=0.05) if (with_motion and i > 0) else None
        t0 = time.perf_counter()
        last_gf = p.process_geometry_frame(observation(l, r, ts=ts, hints=hints))
        elapsed = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            samples.append(elapsed)
    return samples, last_gf, p


def section_main_benchmark(n_warmup: int = 30, n_iters: int = 300, n_trials: int = 5) -> Dict[str, Any]:
    """Authoritative synchronous benchmark of process_geometry_frame().

    MEASUREMENT-QUALITY NOTE: this machine is a shared 4C/8T laptop that was
    NOT idle during the audit (see load_average below). Wall-clock latency is
    therefore reported three ways rather than one:

      * pooled_*      — every sample, exactly as observed under real load.
      * per_trial_medians — n_trials independent repeats; their spread shows
        how much of the pooled figure is external contention.
      * contention_robust_* — min / p10 across all samples and the minimum
        trial median, which approximate the uncontended cost because external
        preemption can only ever ADD time, never remove it.
    """
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(20)

    load_before = os.getloadavg()
    trials, last_gf = [], None
    for _ in range(n_trials):
        gc.collect()
        s, last_gf, p_rect = _bench_geometry_frame(cfg, cal, tf, frames, True, n_warmup, n_iters)
        trials.append(s)
    load_after = os.getloadavg()

    pooled = [x for t in trials for x in t]
    trial_medians = [float(np.median(t)) for t in trials]

    gc.collect()
    s_norect, gf_norect, _ = _bench_geometry_frame(cfg, cal, tf, frames, False, n_warmup, n_iters)

    gc.collect()
    p_leg = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    leg = []
    for i in range(n_warmup + n_iters):
        l, r = frames[i % len(frames)]
        t0 = time.perf_counter()
        p_leg.process(l, r, left_timestamp=float(i) * 0.1)
        e = (time.perf_counter() - t0) * 1000.0
        if i >= n_warmup:
            leg.append(e)

    pooled_a = np.asarray(pooled)
    best_median = float(min(trial_medians))
    return {
        "method_benchmarked": "DepthPerceptionPipeline.process_geometry_frame(StereoObservation) -> GeometryFrame",
        "single_pipeline_instance": True,
        "construction_excluded": True,
        "image_resolution": list(cal.image_size),
        "calibration": F.CALIBRATION_PATH,
        "configuration": "qualified full-V1-candidate config (benchmarks/i0_baseline/scenarios.py::latency_scenario)",
        "rectify": True,
        "body_T_camera_left": "identity rotation, translation [0.05, 0.0, 0.02] m",
        "motion_settings": "5-sample MotionHint window per frame, wz = 0.05 rad/s, dt = 0.1 s",
        "temporal_settings": {
            "enable_temporal": True, "enable_temporal_stabilization": True,
            "enable_rotation_compensation": True, "enable_motion_aware_reliability": True,
            "enable_temporal_persistence": True,
            "temporal_max_records": cfg.temporal_max_records,
            "temporal_max_age_s": cfg.temporal_max_age_s,
            "temporal_gap_limit_s": cfg.temporal_gap_limit_s,
        },
        "n_warmup_per_trial": n_warmup,
        "n_iters_per_trial": n_iters,
        "n_trials": n_trials,
        "load_average_before": list(load_before),
        "load_average_after": list(load_after),
        "machine_was_idle": False,
        "AUTHORITATIVE_pooled": stats_ms(pooled),
        "per_trial_medians_ms": trial_medians,
        "trial_median_spread_ms": float(max(trial_medians) - min(trial_medians)),
        "contention_robust": {
            "min_ms": float(pooled_a.min()),
            "p10_ms": float(np.percentile(pooled_a, 10)),
            "best_trial_median_ms": best_median,
            "fps_from_best_trial_median": float(1000.0 / best_median),
            "fps_from_p10": float(1000.0 / float(np.percentile(pooled_a, 10))),
        },
        "supplementary_rectify_false": stats_ms(s_norect),
        "rectify_false_is_slower_note": (
            "rectify=False is SLOWER here, which is data-dependence, not a rectification saving: "
            "the synthetic fixture is built in already-rectified space, so skipping rectification "
            "preserves far more valid disparity (valid fraction 0.60 vs 0.23) and every downstream "
            "geometry stage then processes ~3x more points. The true cost of rectification itself "
            "is measured directly in the stage-by-stage section."
        ),
        "legacy_process_entry_point": stats_ms(leg),
        "output_sanity_rectify_true": {
            "quality_overall": last_gf.quality.overall_state,
            "depth_valid_fraction": float(np.mean(last_gf.depth_map > 0.0)),
            "geometry_metrics_valid_fraction": (
                float(last_gf.geometry_metrics.valid_fraction) if last_gf.geometry_metrics else None
            ),
            "obstacle_points": int(last_gf.obstacle_cloud.points.shape[0]) if last_gf.obstacle_cloud else None,
        },
        "output_sanity_rectify_false": {
            "quality_overall": gf_norect.quality.overall_state,
            "depth_valid_fraction": float(np.mean(gf_norect.depth_map > 0.0)),
            "geometry_metrics_valid_fraction": (
                float(gf_norect.geometry_metrics.valid_fraction) if gf_norect.geometry_metrics else None
            ),
            "obstacle_points": int(gf_norect.obstacle_cloud.points.shape[0]) if gf_norect.obstacle_cloud else None,
        },
    }


# =====================================================================
# 14. Stage-by-stage latency
# =====================================================================
def section_stage_latency(n_warmup: int = 20, n_iters: int = 120) -> Dict[str, Any]:
    """Two independent, cross-checked views of stage cost.

    A) IN-SITU: the pipeline's own DEBUG stage instrumentation, captured by
       a logging handler. Nothing is added to production code.
    B) RE-DRIVEN: for the stages pipeline.py does not itself log, the exact
       same already-shipped functions are called with the exact same
       arguments, outside the pipeline.
    """
    from depth_perception_engine.fusion.result_builder import build_result
    from depth_perception_engine.geometry.reliability import compute_ramp_zone_mask, compute_shadow_zone_mask

    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(20)

    # ---------- A) in-situ, via existing DEBUG instrumentation ----------
    cap = StageLogCapture()
    prev_level = _PIPELINE_LOGGER.level
    prev_prop = _PIPELINE_LOGGER.propagate
    _PIPELINE_LOGGER.addHandler(cap)
    _PIPELINE_LOGGER.setLevel(logging.DEBUG)
    _PIPELINE_LOGGER.propagate = False
    try:
        p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
        totals = []
        for i in range(n_warmup + n_iters):
            l, r = frames[i % len(frames)]
            ts = float(i) * 0.1
            hints = F.motion_hint_window(ts - 0.1, ts, wz=0.05) if i else None
            if i == n_warmup:
                cap.reset()
                cap.enabled = True
            t0 = time.perf_counter()
            p.process_geometry_frame(observation(l, r, ts=ts, hints=hints))
            e = (time.perf_counter() - t0) * 1000.0
            if i >= n_warmup:
                totals.append(e)
        cap.enabled = False
        in_situ = {k: stats_ms(v) for k, v in sorted(cap.samples.items()) if v}
    finally:
        _PIPELINE_LOGGER.removeHandler(cap)
        _PIPELINE_LOGGER.setLevel(prev_level)
        _PIPELINE_LOGGER.propagate = prev_prop

    # ---------- B) re-driven, for the stages pipeline.py does not log ----------
    p2 = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    rect = p2._rectifier
    de = p2._disparity_engine
    dep = p2._depth_estimator
    si = p2._scene_interpreter
    ta = p2._threat_assessor

    def time_it(fn, n=n_iters, warm=10):
        for _ in range(warm):
            fn()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1000.0)
        return stats_ms(out)

    left, right = frames[0]
    L, R = rect.rectify(left, right)
    gray = L if L.ndim == 2 else cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    raw_disp, _ = de.compute_disparity(L, R, left_gray=gray, compute_visualization=False)
    depth = dep.estimate(raw_disp)

    re_driven = {
        "rectification (cv2.remap x2)": time_it(lambda: rect.rectify(left, right)),
        "grayscale (cv2.cvtColor)": time_it(lambda: cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)),
        "SGBM (StereoSGBM.compute)": time_it(
            lambda: de.compute_disparity(L, R, left_gray=gray, compute_visualization=False)),
        "depth (Q reprojection, Z-only)": time_it(lambda: dep.estimate(raw_disp)),
        "shadow-zone reliability mask": time_it(lambda: compute_shadow_zone_mask(
            raw_disp, raw_disp > 0.0,
            lookahead_px=cfg.geometry_shadow_zone_lookahead_px,
            gradient_threshold_px=cfg.geometry_shadow_zone_gradient_threshold_px,
            max_width_px=cfg.geometry_shadow_zone_max_width_px)),
        "ramp-zone reliability mask": time_it(lambda: compute_ramp_zone_mask(
            raw_disp, raw_disp > 0.0,
            window_px=cfg.clearance_ramp_zone_window_px,
            gradient_threshold_px=cfg.clearance_ramp_zone_gradient_threshold_px)),
        "scene interpretation (regions + decision)": time_it(
            lambda: si.decide_navigation(si.analyze(gray, raw_disp, depth))),
        "threat assessment (per-beam clearance)": time_it(
            lambda: ta.assess(depth, raw_disp, reliability_mask=None)),
    }

    # result assembly + GeometryFrame assembly, measured on a real result
    res = p2.process_observation(observation(left, right, ts=99.0))
    re_driven["GeometryFrame assembly (build_geometry_frame)"] = time_it(
        lambda: p2._build_geometry_frame(res))

    combined = dict(in_situ)
    combined.update(re_driven)
    total_median = float(np.median(totals))
    accounted = sum(v["median_ms"] for v in combined.values())

    ranked = sorted(combined.items(), key=lambda kv: kv[1]["median_ms"], reverse=True)
    return {
        "total_median_ms": total_median,
        "in_situ_stages_from_pipeline_own_instrumentation": in_situ,
        "re_driven_stages": re_driven,
        "combined_ranked": [
            {"stage": k, "median_ms": v["median_ms"], "mean_ms": v["mean_ms"],
             "p95_ms": v["p95_ms"], "pct_of_total": 100.0 * v["median_ms"] / total_median}
            for k, v in ranked
        ],
        "accounted_median_sum_ms": accounted,
        "accounted_fraction_of_total": accounted / total_median,
        "dominant_stage": ranked[0][0],
        "dominant_stage_median_ms": ranked[0][1]["median_ms"],
        "dominant_stage_pct": 100.0 * ranked[0][1]["median_ms"] / total_median,
        "second_stage": ranked[1][0],
        "second_stage_median_ms": ranked[1][1]["median_ms"],
        "second_stage_pct": 100.0 * ranked[1][1]["median_ms"] / total_median,
    }


# =====================================================================
# 15. Temporal vs non-temporal cost
# =====================================================================
def section_temporal_vs_non_temporal(n_warmup: int = 20, n_iters: int = 200) -> Dict[str, Any]:
    cal = F.calibration()
    tf = F.body_transform()
    frames = F.scene_sequence(20)

    gc.collect()
    s_temp, gf_temp, _ = _bench_geometry_frame(
        F.qualified_config(), cal, tf, frames, True, n_warmup, n_iters)
    gc.collect()
    s_none, gf_none, _ = _bench_geometry_frame(
        F.non_temporal_config(), cal, tf, frames, True, n_warmup, n_iters)

    def summarize(gf: GeometryFrame) -> Dict[str, Any]:
        return {
            "temporal_consistency": gf.temporal_consistency.state if gf.temporal_consistency else None,
            "temporal_stabilization": gf.temporal_stabilization.state if gf.temporal_stabilization else None,
            "rotation_compensation_status": gf.rotation_compensation_status,
            "motion_aware_reliability": gf.motion_aware_reliability.state if gf.motion_aware_reliability else None,
            "temporal_persistence": gf.temporal_persistence.state if gf.temporal_persistence else None,
            "quality_overall": gf.quality.overall_state,
            "quality_temporal_consistency_state": gf.quality.temporal_consistency_state,
            "quality_motion_reliability_state": gf.quality.motion_reliability_state,
            "quality_persistence_state": gf.quality.persistence_state,
            "degradation_reasons": list(gf.quality.degradation_reasons),
            "depth_valid_fraction": float(np.mean(gf.depth_map > 0.0)),
        }

    st, sn = stats_ms(s_temp), stats_ms(s_none)
    return {
        "path": "same authoritative process_geometry_frame() path in BOTH arms; only supported config flags differ",
        "A_temporal_enabled_ms": st,
        "B_temporal_disabled_ms": sn,
        "delta_median_ms": st["median_ms"] - sn["median_ms"],
        "delta_mean_ms": st["mean_ms"] - sn["mean_ms"],
        "delta_p95_ms": st["p95_ms"] - sn["p95_ms"],
        "temporal_share_of_median_pct": 100.0 * (st["median_ms"] - sn["median_ms"]) / st["median_ms"],
        "output_A_temporal": summarize(gf_temp),
        "output_B_non_temporal": summarize(gf_none),
        "raster_geometry_identical": bool(np.array_equal(gf_temp.depth_map, gf_none.depth_map)),
    }


# =====================================================================
# 16. Resolution scaling
# =====================================================================
def section_resolution_scaling(n_warmup: int = 15, n_iters: int = 80) -> Dict[str, Any]:
    from depth_perception_engine.geometry.reliability import compute_shadow_zone_mask

    cal = F.calibration()
    tf = F.body_transform()
    cfg = F.qualified_config()
    base_frames = F.scene_sequence(10)

    results = {}
    for factor in (1, 2):
        c = cal if factor == 1 else F.scaled_calibration(cal, factor)
        frames = base_frames if factor == 1 else [F.upscale_pair(l, r, factor) for l, r in base_frames]
        gc.collect()
        s, gf, p = _bench_geometry_frame(cfg, c, tf, frames, True, n_warmup, n_iters)

        # isolate SGBM at this resolution using the pipeline's own engines
        L, R = p._rectifier.rectify(*frames[0])
        gray = L if L.ndim == 2 else cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
        for _ in range(5):
            p._disparity_engine.compute_disparity(L, R, left_gray=gray, compute_visualization=False)
        sg = []
        for _ in range(40):
            t0 = time.perf_counter()
            p._disparity_engine.compute_disparity(L, R, left_gray=gray, compute_visualization=False)
            sg.append((time.perf_counter() - t0) * 1000.0)
        rd, _ = p._disparity_engine.compute_disparity(L, R, left_gray=gray, compute_visualization=False)
        rc = []
        for _ in range(40):
            t0 = time.perf_counter()
            p._rectifier.rectify(*frames[0])
            rc.append((time.perf_counter() - t0) * 1000.0)

        results[f"{c.image_size[0]}x{c.image_size[1]}"] = {
            "scale_factor": factor,
            "calibration": "hardware fixture (QUALIFIED)" if factor == 1
                           else "exactly-derived %dx scaling of the same rig (NOT a qualified configuration)" % factor,
            "pixels": int(c.image_size[0] * c.image_size[1]),
            "total_ms": stats_ms(s),
            "sgbm_ms": stats_ms(sg),
            "rectification_ms": stats_ms(rc),
            "geometry_and_other_median_ms": float(np.median(s)) - float(np.median(sg)) - float(np.median(rc)),
            "quality_overall": gf.quality.overall_state,
            "depth_valid_fraction": float(np.mean(gf.depth_map > 0.0)),
        }

    k1, k2 = list(results.keys())
    r1, r2 = results[k1], results[k2]
    results["scaling"] = {
        "pixel_ratio": r2["pixels"] / r1["pixels"],
        "total_median_ratio": r2["total_ms"]["median_ms"] / r1["total_ms"]["median_ms"],
        "sgbm_median_ratio": r2["sgbm_ms"]["median_ms"] / r1["sgbm_ms"]["median_ms"],
        "rectification_median_ratio": r2["rectification_ms"]["median_ms"] / r1["rectification_ms"]["median_ms"],
        "geometry_other_ratio": r2["geometry_and_other_median_ms"] / r1["geometry_and_other_median_ms"],
        "fps_at_qualified": r1["total_ms"]["fps_from_median"],
        "fps_at_derived_2x": r2["total_ms"]["fps_from_median"],
    }
    return results


# =====================================================================
# 17. Repeated-run stability
# =====================================================================
def section_stability(n_frames: int = 400) -> Dict[str, Any]:
    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(20)

    gc.collect()
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    ids0 = {
        "rectifier": id(p._rectifier), "disparity_engine": id(p._disparity_engine),
        "depth_estimator": id(p._depth_estimator), "point_cloud_builder": id(p._point_cloud_builder),
        "scene_interpreter": id(p._scene_interpreter), "threat_assessor": id(p._threat_assessor),
        "temporal_history": id(p._temporal_history),
        "persistence_tracker": id(p._temporal_persistence_tracker),
    }
    rss_start = rss_mb()
    samples, trace = [], []
    for i in range(n_frames):
        l, r = frames[i % len(frames)]
        ts = float(i) * 0.1
        hints = F.motion_hint_window(ts - 0.1, ts, wz=0.05) if i else None
        t0 = time.perf_counter()
        gf = p.process_geometry_frame(observation(l, r, ts=ts, hints=hints))
        samples.append((time.perf_counter() - t0) * 1000.0)
        if i % 50 == 0 or i == n_frames - 1:
            tp = p._temporal_persistence_tracker
            trace.append({
                "frame": i,
                "rss_mb": rss_mb(),
                "history_len": len(p.temporal_history),
                "persistence_grid_shape": jsonable(tp._grid_shape) if tp is not None else None,
                "persistence_support_max": int(tp._support_count.max()) if tp is not None and tp._support_count is not None else None,
                "threat_ema_len": int(np.asarray(p._threat_assessor._ema_dist).size),
                "obstacle_points": int(gf.obstacle_cloud.points.shape[0]) if gf.obstacle_cloud else None,
                "opening_evidence_len": len(gf.opening_evidence) if gf.opening_evidence is not None else None,
                "boundary_evidence_len": len(gf.boundary_evidence) if gf.boundary_evidence is not None else None,
                "quality_overall": gf.quality.overall_state,
            })
    rss_end = rss_mb()
    gc.collect()
    rss_after_gc = rss_mb()
    ids1 = {
        "rectifier": id(p._rectifier), "disparity_engine": id(p._disparity_engine),
        "depth_estimator": id(p._depth_estimator), "point_cloud_builder": id(p._point_cloud_builder),
        "scene_interpreter": id(p._scene_interpreter), "threat_assessor": id(p._threat_assessor),
        "temporal_history": id(p._temporal_history),
        "persistence_tracker": id(p._temporal_persistence_tracker),
    }

    a = np.asarray(samples)
    q = max(1, len(a) // 4)
    quarters = [float(np.median(a[i * q:(i + 1) * q])) for i in range(4)]
    return {
        "n_frames": n_frames,
        "crashed": False,
        "latency_ms": stats_ms(samples),
        "quarter_medians_ms": quarters,
        "latency_drift_last_vs_first_quarter_ms": quarters[-1] - quarters[0],
        "latency_drift_pct": 100.0 * (quarters[-1] - quarters[0]) / quarters[0],
        "rss_mb_start": rss_start,
        "rss_mb_end": rss_end,
        "rss_mb_after_gc": rss_after_gc,
        "rss_growth_mb": rss_end - rss_start,
        "object_identity_stable": {k: ids0[k] == ids1[k] for k in ids0},
        "temporal_history_bounded_at_max_records": len(p.temporal_history) <= cfg.temporal_max_records,
        "final_history_len": len(p.temporal_history),
        "max_records_config": cfg.temporal_max_records,
        "trace": trace,
        "frames_processed_counter": p.health().frames_processed,
    }


# =====================================================================
# 20. Native compute / GIL characterization
# =====================================================================
def section_native_vs_python() -> Dict[str, Any]:
    """Classify stages as native-heavy / Python-heavy / mixed.

    Empirical method — NO concurrency is introduced. cv2.setNumThreads() is
    varied and each stage re-timed. A stage that speeds up when OpenCV is
    allowed more threads is executing inside OpenCV's own native parallel
    region, which by construction runs outside the Python interpreter (and
    therefore does not hold the GIL for that work). A stage whose time is
    unchanged is either single-threaded native or Python/NumPy-bound; those
    are separated by static inspection of the implementing module.
    """
    from depth_perception_engine.geometry.reliability import compute_ramp_zone_mask, compute_shadow_zone_mask

    cal = F.calibration()
    cfg = F.qualified_config()
    tf = F.body_transform()
    frames = F.scene_sequence(4)
    p = DepthPerceptionPipeline(cfg, cal, rectify=True, body_T_camera_left=tf)
    left, right = frames[0]
    L, R = p._rectifier.rectify(left, right)
    gray = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    raw_disp, _ = p._disparity_engine.compute_disparity(L, R, left_gray=gray, compute_visualization=False)
    depth = p._depth_estimator.estimate(raw_disp)
    origin = tf.translation
    cloud = p._point_cloud_builder.build(raw_disp, timestamp=0.0)
    from depth_perception_engine.geometry.rigid_transform import transform_point_cloud
    cloud_b = transform_point_cloud(cloud, tf)
    from depth_perception_engine.geometry.obstacle_extractor import build_obstacle_cloud
    from depth_perception_engine.geometry.surface import build_surface_evidence
    from depth_perception_engine.geometry.boundary import build_boundary_evidence

    stages = {
        "rectification (cv2.remap)": lambda: p._rectifier.rectify(left, right),
        "grayscale (cv2.cvtColor)": lambda: cv2.cvtColor(L, cv2.COLOR_BGR2GRAY),
        "SGBM (cv2.StereoSGBM.compute)": lambda: p._disparity_engine.compute_disparity(
            L, R, left_gray=gray, compute_visualization=False),
        "depth (NumPy Q reprojection)": lambda: p._depth_estimator.estimate(raw_disp),
        "point cloud (NumPy)": lambda: p._point_cloud_builder.build(raw_disp, timestamp=0.0),
        "body transform (NumPy einsum)": lambda: transform_point_cloud(cloud, tf),
        "obstacle cloud (NumPy)": lambda: build_obstacle_cloud(
            cloud_b, origin, min_range_m=cfg.obstacle_min_range_m,
            max_range_m=cfg.obstacle_max_range_m, stride=cfg.geometry_sampling_stride,
            reliability_mask=None),
        "shadow-zone mask (NumPy)": lambda: compute_shadow_zone_mask(
            raw_disp, raw_disp > 0.0, lookahead_px=cfg.geometry_shadow_zone_lookahead_px,
            gradient_threshold_px=cfg.geometry_shadow_zone_gradient_threshold_px,
            max_width_px=cfg.geometry_shadow_zone_max_width_px),
        "ramp-zone mask (NumPy)": lambda: compute_ramp_zone_mask(
            raw_disp, raw_disp > 0.0, window_px=cfg.clearance_ramp_zone_window_px,
            gradient_threshold_px=cfg.clearance_ramp_zone_gradient_threshold_px),
        "surface evidence (NumPy + per-cell Python loop)": lambda: build_surface_evidence(
            cloud_b, origin, grid_rows=cfg.surface_grid_rows, grid_cols=cfg.surface_grid_cols,
            min_support_count=cfg.surface_min_support_count, reliability_mask=None),
        "boundary evidence (NumPy + per-cell Python loop)": lambda: build_boundary_evidence(
            depth, FrameId.CAMERA_OPTICAL_LEFT, grid_rows=cfg.boundary_grid_rows,
            grid_cols=cfg.boundary_grid_cols, min_support_count=cfg.boundary_min_support_count,
            depth_step_threshold_m=cfg.boundary_depth_step_threshold_m,
            orientation_change_threshold_rad=cfg.boundary_orientation_change_threshold_rad,
            surface_evidence=None, surface_grid_rows=cfg.surface_grid_rows,
            surface_grid_cols=cfg.surface_grid_cols, reliability_mask=None,
            min_confirmation_support_fraction=cfg.boundary_min_confirmation_support_fraction),
        "scene interpretation": lambda: p._scene_interpreter.decide_navigation(
            p._scene_interpreter.analyze(gray, raw_disp, depth)),
        "threat assessment": lambda: p._threat_assessor.assess(depth, raw_disp, reliability_mask=None),
    }

    def timed(fn, n=60, warm=8):
        for _ in range(warm):
            fn()
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            out.append((time.perf_counter() - t0) * 1000.0)
        return float(np.median(out))

    original_threads = cv2.getNumThreads()
    per_stage = {}
    # INTERLEAVED A/B/A/B per stage, with both arms warmed before either is
    # timed. Measuring all of arm A and then all of arm B would give arm B a
    # warm-cache advantage and manufacture a fake speedup on every stage.
    one_r, many_r = {k: [] for k in stages}, {k: [] for k in stages}
    try:
        for k, fn in stages.items():
            cv2.setNumThreads(1)
            timed(fn, n=5, warm=8)
            cv2.setNumThreads(original_threads)
            timed(fn, n=5, warm=8)
            for _ in range(3):
                cv2.setNumThreads(1)
                one_r[k].append(timed(fn, n=25, warm=3))
                cv2.setNumThreads(original_threads)
                many_r[k].append(timed(fn, n=25, warm=3))
    finally:
        cv2.setNumThreads(original_threads)
    one = {k: float(np.median(v)) for k, v in one_r.items()}
    many = {k: float(np.median(v)) for k, v in many_r.items()}

    for k in stages:
        speedup = one[k] / many[k] if many[k] > 0 else 1.0
        if speedup >= 1.35:
            cls = "native-heavy (OpenCV internal parallel region; runs outside the interpreter)"
        elif speedup >= 1.10:
            cls = "mixed"
        else:
            cls = "single-threaded (native single-thread or NumPy/Python bound)"
        per_stage[k] = {
            "median_ms_cv2_threads_1": one[k],
            "median_ms_cv2_threads_%d" % original_threads: many[k],
            "opencv_thread_speedup": speedup,
            "classification": cls,
        }

    # Static inspection: where do Python-level loops actually exist?
    src_root = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")
    import ast
    loops = {}
    for dirpath, _d, files in os.walk(src_root):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            with open(path, "r", encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read(), filename=path)
                except SyntaxError:
                    continue
            n = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
            if n:
                loops[os.path.relpath(path, _REPO_ROOT)] = n

    return {
        "method": "cv2.setNumThreads(1) vs cv2.setNumThreads(%d), INTERLEAVED A/B/A/B per stage "
                  "(3 repeats each, median of repeats), same process, no concurrency added"
                  % original_threads,
        "numpy_blas_backend": "scipy-openblas 0.3.33 (DYNAMIC_ARCH, MAX_THREADS=64); no OMP/OPENBLAS "
                              "thread env var is set, so NumPy matmul may thread independently of "
                              "cv2.setNumThreads()",
        "per_arm_repeats_ms": {"cv2_threads_1": one_r, "cv2_threads_%d" % original_threads: many_r},
        "opencv_threads_default": original_threads,
        "per_stage": per_stage,
        "python_loop_counts_by_production_module": dict(sorted(loops.items())),
        "native_calls_used": [
            "cv2.remap (rectification)", "cv2.cvtColor (grayscale)",
            "cv2.StereoSGBM.compute (disparity)", "cv2.medianBlur / cv2.Sobel (post-processing)",
        ],
        "reprojectImageTo3D_used": False,
        "reprojectImageTo3D_note": (
            "DepthEstimator.estimate() deliberately does NOT call cv2.reprojectImageTo3D — it "
            "computes the Z channel directly in NumPy (see depth_estimator.py's docstring)."
        ),
    }


# =====================================================================
# 21. DPE vs frozen NPE — theoretical provider-overlap bounds
# =====================================================================
NPE_FROZEN = {
    "commit": "2dac0799752e16c2b8767cd000b236f720489315",
    "version": "2.0.0",
    "yolox_median_ms": 10.1,
    "roadseg_median_ms": 78.0,
    "combined_median_ms": 95.3,
    "combined_fps": 10.5,
    "runtime": "ONNX Runtime, CPU execution provider",
}


def section_dpe_vs_npe(dpe_median_ms: float) -> Dict[str, Any]:
    npe = NPE_FROZEN["combined_median_ms"]
    seq = dpe_median_ms + npe
    floor = max(dpe_median_ms, npe)
    saving = seq - floor
    return {
        "npe_frozen": NPE_FROZEN,
        "dpe_median_ms": dpe_median_ms,
        "sequential_provider_wall_ms": seq,
        "sequential_provider_wall_fps": 1000.0 / seq,
        "theoretical_independent_floor_ms": floor,
        "theoretical_independent_floor_fps": 1000.0 / floor,
        "theoretical_maximum_overlap_saving_ms": saving,
        "theoretical_maximum_overlap_saving_pct": 100.0 * saving / seq,
        "dpe_fps_alone": 1000.0 / dpe_median_ms,
        "npe_fps_alone": 1000.0 / npe,
        "bound_is_theoretical_only": True,
        "note": (
            "These are arithmetic bounds on a perfectly overlapped two-provider schedule with "
            "zero contention and zero join cost. No concurrent execution was measured in D1."
        ),
    }


# =====================================================================
# driver
# =====================================================================
def run_all() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    steps = [
        ("environment", section_environment),
        ("identity", section_identity),
        ("timestamps", section_timestamps),
        ("geometry_frame_contract", section_geometry_frame_contract),
        ("mutable_state", section_mutable_state),
        ("reentrancy", section_reentrancy_evidence),
        ("construction", section_construction),
        ("reset_close", section_reset_close),
        ("motion_path", section_motion_path),
        ("unit_contract", section_unit_contract),
        ("discontinuity", section_discontinuity),
        ("failure_semantics", section_failure_semantics),
        ("main_benchmark", section_main_benchmark),
        ("stage_latency", section_stage_latency),
        ("temporal_vs_non_temporal", section_temporal_vs_non_temporal),
        ("resolution_scaling", section_resolution_scaling),
        ("stability", section_stability),
        ("native_vs_python", section_native_vs_python),
    ]
    for name, fn in steps:
        t0 = time.perf_counter()
        print(f"[D1] {name} ...", flush=True)
        out[name] = fn()
        print(f"[D1] {name} done in {time.perf_counter() - t0:.1f}s", flush=True)

    dpe_median = out["main_benchmark"]["contention_robust"]["best_trial_median_ms"]
    out["dpe_vs_npe"] = section_dpe_vs_npe(dpe_median)
    out["dpe_vs_npe_using_pooled_median"] = section_dpe_vs_npe(
        out["main_benchmark"]["AUTHORITATIVE_pooled"]["median_ms"]
    )
    return out


def main() -> None:
    results = run_all()
    path = os.path.join(RESULTS_DIR, "d1_execution_audit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(jsonable(results), fh, indent=2, sort_keys=False)
    print(f"[D1] wrote {path}")


if __name__ == "__main__":
    main()
