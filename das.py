import numpy as np
from numba import njit, prange
from numpy import typing as npt


@njit(parallel=True, fastmath=True)
def _calculate_das_map_numba(fft_signals, steering_vectors):
  num_points, num_freq, num_mics = steering_vectors.shape
  output = np.zeros((num_points, num_freq), dtype=np.complex64)

  for p_idx in prange(num_points):
    for f_idx in range(num_freq):
      sum_val = 0.0 + 0.0j
      for m_idx in range(num_mics):
        sig_val = fft_signals[f_idx, m_idx]
        steer_val = steering_vectors[p_idx, f_idx, m_idx]
        sum_val += sig_val * steer_val
      output[p_idx, f_idx] = sum_val / num_mics
  return output

class DAS:
  _mic_positions: npt.NDArray
  _speed_of_sound: float
  _frequencies: npt.NDArray
  _steering_vectors_array: npt.NDArray

  def __init__(
    self,
    mic_positions: npt.NDArray,
    speed_of_sound: float,
    frequencies: npt.NDArray
  ):
    self._mic_positions = mic_positions
    self._speed_of_sound = speed_of_sound
    self._frequencies = frequencies
    self.fft_signals = None

  def prepare_chunk(self, fft_signals):
    self.fft_signals = fft_signals

  def prepare_probe_points(self, points):
    num_points = len(points)
    num_freq = len(self._frequencies)
    num_mics = len(self._mic_positions)
    
    self._steering_vectors_array = np.zeros(
      (num_points, num_freq, num_mics),
      dtype=np.complex64
    )

    for i, (elevation, azimuth) in enumerate(points):
      az_rad = np.radians(azimuth)
      el_rad = np.radians(elevation)

      direction_vector = np.array([
        np.sin(az_rad) * np.cos(el_rad),
        np.sin(el_rad),
        np.cos(az_rad) * np.cos(el_rad)
      ])

      distances = np.dot(self._mic_positions, direction_vector)
      time_delays = distances / self._speed_of_sound

      omega = 2.0 * np.pi * self._frequencies[:, np.newaxis]
      tau = time_delays[np.newaxis, :]

      self._steering_vectors_array[i] = np.exp(-1j * omega * tau).astype(np.complex64)

  def calculate_grid(self):
    num_mics = len(self._mic_positions)
    fft_signals_sliced = self.fft_signals[:, :num_mics].astype(np.complex64)
    
    return _calculate_das_map_numba(
      fft_signals_sliced,
      self._steering_vectors_array
    )
