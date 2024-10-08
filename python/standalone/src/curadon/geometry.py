import numpy as np
import numpy.typing as npt

from typing import List, Tuple, Optional, Any, Union
from types import ModuleType

from . import backend as _C


def is_convertible_to_float(val: Any):
    try:
        float(val)
        return True
    except ValueError:
        return False
    except TypeError:
        return False



def geom_to_str(geom):
    parameters = []
    parameters.append("-----")
    parameters.append("Geometry parameters")
    parameters.append(f"Distance from source to detector (DSD) = {geom.DSD}")
    parameters.append(f"Distance from source to origin (DSO) = {geom.DSO}")
    if isinstance(geom, FanGeometry):
        parameters.append(f"Projections angles = {np.rad2deg(geom.angles)}")
    parameters.append("-----")
    parameters.append("Detector parameters")
    parameters.append(f"Number of pixels (det_shape) = {geom.det_shape}")
    parameters.append(f"Size of each pixel (det_spacing) = {geom.det_spacing}")
    parameters.append(
        f"Total size of the detector (det_size) = {geom.det_shape * geom.det_spacing}")
    parameters.append(
        f"Detector rotation (det_rotation) = {geom.det_rotation}")

    parameters.append(f"Offset of detector (det_offset) = {geom.det_offset}")
    parameters.append("-----")
    parameters.append("Image parameters")
    parameters.append(f"Number of voxels (vol_shape) = {geom.vol_shape}")
    parameters.append(f"Size of each voxel (vol_spacing) = {geom.vol_spacing}")
    parameters.append(
        f"Total size of the image (vol_size) = {geom.vol_shape * geom.vol_spacing}")
    parameters.append(
        f"Offset of image from origin (vol_offset) = {geom.vol_offset}")

    # parameters.append("-----")
    # parameters.append(f"Centre of rotation correction (COR) = {geom.COR}")
    return "\n".join(parameters)


class FanGeometry:
    def to_vec(self):
        """
        Create an ASTRA-like vector description of a given FanGeometry
        """
        vecs = []
        # This is now implemented both here and in C++, I think that should be
        # moved to one source of truth, as mismatches would be terrible to
        # debug...
        for (alpha, dsd, dso) in zip(self.angles, self.DSD, self.DSO):
            src = np.array([0, dso])
            dod = dsd - dso
            det = np.array([0, -dod])
            u = np.array([1, 0])

            cs = np.cos(alpha)
            sn = np.sin(alpha)

            R = np.asarray([[cs, sn], [-sn, cs]])

            src = np.dot(R, src)
            det = np.dot(R, det)
            u = np.dot(R, u)

            vecs.append([*src.tolist(), *det.tolist(), *u.tolist()])

        return vecs




    @staticmethod
    def from_vec(vol_shape: Tuple[int, int], det_shape: int, vecs: List[List[float]], **kwargs):
        """
        Create FanGeometry from an ASTRA-like vector description.

        Parameters
        ----------
            det_shape : int
                Number of pixels of the flat-panel detector
            vecs : List[List[float]]
                A list of vectors of length 6 describing the geometry. The 6
                entries describe `[source_pos.x, source_pos.y, det_pos.x,
                det_pos.y, u.x, u.y], where `source_pos` is the position of the
                source, `det_pos` is the center of the detector, and `u` is the
                direction vector from center of detector pixel 0 to detector
                pixel 1
            **kwargs :
                Arguments passed on to the constructor of FanGeometry

        """
        vecs = np.asarray(vecs)

        dsd = []
        dso = []
        angles = []
        det_spacing = np.linalg.norm(np.asarray(vecs[0][-2:]))
        for v in vecs:
            sx, sy, dx, dy, ux, uy = v

            src = np.asarray([sx, sy])
            det = np.asarray([dx, dy])
            u = np.asarray([ux, uy])

            src2det = src - det
            dsd.append(np.linalg.norm(src2det))
            dso.append(np.linalg.norm(src))

            # atan2 gives the angle of the line between source and detector
            # ensure no rotation is given if src is at [0, dso], and detector
            # at [0, dso - dsd], and rotation is clockwise
            angles.append(np.atan2(*src2det) % (2*np.pi))

            # TODO: check how astra handles this? I don't think it supports changing detector spacing
            # TODO: this is also more than the spacing, it also indicates start of detector,
            #       which is currently hard-coded in curadon
            if not np.isclose(det_spacing, np.linalg.norm(u)):
                raise AttributeError("Detector must have equal spacing for all projections")

        return FanGeometry(DSD=dsd, DSO=dso, angles=angles, vol_shape=vol_shape,
                           det_shape=det_shape, det_spacing=det_spacing, **kwargs)


    def __init__(self, DSD: Union[float, List[float]],
                 DSO: Union[float, List[float]],
                 angles: List[float],
                 vol_shape: Tuple[int, int],
                 det_shape: int,
                 det_spacing: Optional[float] = None,
                 det_offset: Optional[float] = None,
                 det_rotation: Optional[float] = None,
                 vol_spacing: Optional[Tuple[float, float]] = None,
                 vol_offset: Optional[Tuple[float, float]] = None,
                 device: int = 0,
                 vol_prec: int = 32,
                 sino_prec: int = 32,
                 ):
        # Check angles
        self.angles = np.asarray(angles)
        if self.angles.ndim != 1:
            raise AttributeError(
                f"Angles must be 1D array, got {str(angles.shape)}")
        self.nangles = self.angles.shape[0]

        if is_convertible_to_float(DSD) is True:
            self.DSD = np.full(self.nangles, DSD)
        else:
            self.DSD = np.asarray(DSD)
            if self.DSD.ndim != 1 or self.DSD.size != self.nangles:
                raise AttributeError(f"DSD must be a single float, or a list of floats nangles entries (is {self.DSD.size}, expected {self.nangles})")

        if is_convertible_to_float(DSO) is True:
            self.DSO = np.full(self.nangles, DSD)
        else:
            self.DSO = np.asarray(DSO)
            if self.DSO.ndim != 1 or self.DSO.size != self.nangles:
                raise AttributeError(f"DSO must be a single float, or a list of floats nangles entries (is {self.DSO.size}, expected {self.nangles})")

        # Check vol_shape
        self.vol_shape = np.asarray(vol_shape)
        if self.vol_shape.shape != (2,):
            raise AttributeError(
                f"vol_shape must be two values, got {str(self.vol_shape.shape)}")

        # Check det_shape
        if is_convertible_to_float(det_shape) is False:
            raise AttributeError(
                f"det_shape must to be float, got {type(det_shape)}")
        self.det_shape = int(det_shape)

        # Check det_spacing
        if det_spacing is not None:
            if is_convertible_to_float(det_spacing) is False:
                raise TypeError(
                    f"Expected det_spacing to be float, got {type(det_spacing)}")
            self.det_spacing = float(det_spacing)
        else:
            self.det_spacing = float(1.)

        # Check det_offset
        if det_offset is not None:
            if is_convertible_to_float(det_offset) is False:
                raise TypeError(
                    f"Expected det_offset to be float, got {type(det_offset)}")
            self.det_offset = float(det_offset)
        else:
            self.det_offset = float(0)

        # Check det_rotation
        if det_rotation is not None:
            if is_convertible_to_float(det_rotation) is False:
                raise TypeError(
                    f"Expected det_rotation to be float, got {type(det_rotation)}")
            self.det_rotation = float(det_rotation)
        else:
            self.det_rotation = float(0)

        if vol_spacing is not None:
            self.vol_spacing = np.asarray(vol_spacing)
            if self.vol_spacing.shape != (2,):
                raise AttributeError(
                    f"vol_spacing must be two values, got {str(self.vol_spacing.shape)}")
        else:
            self.vol_spacing = np.ones(self.vol_shape.shape)

        if vol_offset is not None:
            self.vol_offset = np.asarray(vol_offset)
            if self.vol_offset.shape != (2,):
                raise AttributeError(
                    f"vol_offset must be two values, got {str(self.vol_offset.shape)}")
        else:
            self.vol_offset = np.zeros(self.vol_shape.shape)

        self.plan = _C.plan_2d(device, vol_prec, self.vol_shape, self.vol_spacing, self.vol_offset,
                               sino_prec, self.det_shape, self.det_spacing, self.det_offset, self.DSO, self.DSD,
                               self.angles, self.det_rotation, 0) # COR currently still ignored

    def sinogram_shape(self):
        return (self.nangles, self.det_shape)

    def __str__(self):
        return geom_to_str(self)


class ConeGeometry:
    def __init__(self, DSD: float,
                 DSO: float,
                 angles: Union[List[float], List[List[float]]],
                 vol_shape: Tuple[int, int, int],
                 det_shape: Tuple[int, int],
                 det_spacing: Optional[Tuple[float, float]] = None,
                 det_offset: Optional[Tuple[float, float]] = None,
                 det_rotation: Optional[Tuple[float, float, float]] = None,
                 vol_spacing: Optional[Tuple[float, float, float]] = None,
                 vol_offset: Optional[Tuple[float, float, float]] = None,
                 COR: Optional[float] = None):
        if is_convertible_to_float(DSD) is False:
            raise TypeError(f"Expected DSD to be float, got {type(DSD)}")
        self.DSD = float(DSD)

        if is_convertible_to_float(DSO) is False:
            raise TypeError(f"Expected DSO to be float, got {type(DSO)}")
        self.DSO = float(DSO)

        self.det_shape = np.asarray(det_shape)
        if self.det_shape.shape != (2, ):
            raise AttributeError(
                f"Detector shape must be two values, got {str(self.det_shape.shape)}")

        self.vol_shape = np.asarray(vol_shape)
        if self.vol_shape.shape != (3, ):
            raise AttributeError(
                f"Volume shape must be three values, got {str(self.vol_shape.shape)}")

        self.angles = np.asarray(angles)
        if self.angles.ndim == 1:
            self.nangles = self.angles.shape[0]
            zeros = np.zeros((self.nangles,), dtype=np.float32)
            self.angles = np.vstack((self.angles, zeros, zeros))
        elif self.angles.ndim == 2:
            if self.angles.shape[0] != 3:
                raise AttributeError(
                    f"Angles must either be a 1D array, or a 2D array of shape (3, nangles), got {str(self.angles.shape)}")

            self.nangles = self.angles.shape[1]
        else:
            raise AttributeError(
                f"Angles must either be a 1D array, or a 2D array of shape (3, nangles), got {str(self.angles.shape)}")

        if det_spacing is not None:
            self.det_spacing = np.asarray(det_spacing)
            if self.det_spacing.shape != (2, ):
                raise AttributeError(
                    f"det_spacing must be two values, got {str(self.det_spacing.shape)}")
        else:
            self.det_spacing = np.ones(self.det_shape.shape)

        if det_offset is not None:
            self.det_offset = np.asarray(det_offset)
            if self.det_offset.shape != (2, ):
                raise AttributeError(
                    f"det_offset must be two values, got {str(self.det_offset.shape)}")
        else:
            self.det_offset = np.zeros(self.det_shape.shape)

        if det_rotation is not None:
            self.det_rotation = np.asarray(det_rotation)
            if self.det_rotation.shape != (3, ):
                raise AttributeError(
                    f"det_rotation must be three values, got {str(self.det_rotation.shape)}")
        else:
            self.det_rotation = np.zeros((3, ))

        # Check COR
        if COR is not None:
            if is_convertible_to_float(COR) is False:
                raise TypeError(
                    f"Expected COR to be float, got {type(COR)}")
            self.COR = float(COR)
        else:
            self.COR = float(0)

        if vol_spacing is not None:
            self.vol_spacing = np.asarray(vol_spacing)
            if self.vol_spacing.shape != (3, ):
                raise AttributeError(
                    f"vol_spacing must be three values, got {str(self.vol_spacing.shape)}")
        else:
            self.vol_spacing = np.ones(self.vol_shape.shape)

        self.vol_size = self.vol_shape * self.vol_spacing
        if vol_offset is not None:
            self.vol_offset = np.asarray(vol_offset)
            if self.vol_offset.shape != (3, ):
                raise AttributeError(
                    f"vol_offset must be three values, got {str(self.vol_offset.shape)}")
        else:
            self.vol_offset = np.zeros(self.vol_shape.shape)

    def sinogram_shape(self):
        return (self.nangles, *self.det_shape)

    def __str__(self):
        return geom_to_str(self)
