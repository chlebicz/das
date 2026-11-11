import numpy as np

from das import DAS


def simulate_plane_wave(
  mic_positions: np.ndarray,
  speed_of_sound: float,
  sample_rate: int,
  chunk_size: int,
  frequency: float,
  elevation_deg: float,
  azimuth_deg: float,
) -> np.ndarray:
  """Generate a synthetic multi-channel time-domain signal for a far-field source."""
  az_rad = np.radians(azimuth_deg)
  el_rad = np.radians(elevation_deg)

  direction_vector = np.array([
    np.sin(az_rad) * np.cos(el_rad),
    np.sin(el_rad),
    np.cos(az_rad) * np.cos(el_rad),
  ])

  distances = np.dot(mic_positions, direction_vector)
  time_delays = distances / speed_of_sound

  t = np.arange(chunk_size)[:, np.newaxis] / sample_rate
  signals = np.cos(2.0 * np.pi * frequency * (t + time_delays[np.newaxis, :]))
  return signals

def main():
  # 8 mic circular array
  mic_positions = np.array([
    [ 0.050016725, -0.120751055,  0.0],
    [ 0.120751055, -0.050016725,  0.0],
    [ 0.120751055,  0.050016725,  0.0],
    [ 0.050016725,  0.120751055,  0.0],
    [-0.050016725,  0.120751055,  0.0],
    [-0.120751055,  0.050016725,  0.0],
    [-0.120751055, -0.050016725,  0.0],
    [-0.050016725, -0.120751055,  0.0]
  ])

  sample_rate = 48000
  chunk_size = 2048
  speed_of_sound = 343.0
  min_freq = 200.0
  max_freq = 4000.0

  all_freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)
  freq_mask = (all_freqs >= min_freq) & (all_freqs <= max_freq)
  frequencies = all_freqs[freq_mask]

  das = DAS(
    mic_positions=mic_positions,
    speed_of_sound=speed_of_sound,
    frequencies=frequencies,
  )

  elevation_min = -70
  elevation_max = 71
  elevation_step = 5

  azimuth_min = -70
  azimuth_max = 71
  azimuth_step = 5

  elevations = np.arange(elevation_min, elevation_max, elevation_step)
  azimuths = np.arange(azimuth_min, azimuth_max, azimuth_step)
  el_grid, az_grid = np.meshgrid(elevations, azimuths)
  probe_points = np.vstack([el_grid.ravel(), az_grid.ravel()]).T

  das.prepare_probe_points(probe_points)

  true_elevation = 15.0
  true_azimuth = -25.0
  source_freq = 1000.0

  time_signals = simulate_plane_wave(
    mic_positions=mic_positions,
    speed_of_sound=speed_of_sound,
    sample_rate=sample_rate,
    chunk_size=chunk_size,
    frequency=source_freq,
    elevation_deg=true_elevation,
    azimuth_deg=true_azimuth,
  )

  fft_signals = np.fft.rfft(time_signals, axis=0)[freq_mask, :]

  das.prepare_chunk(fft_signals)

  beamformed_grid = das.calculate_grid()

  power_linear = np.sum(np.abs(beamformed_grid) ** 2 * 2, axis=1) / (chunk_size ** 2)
  power_db = 10.0 * np.log10(power_linear + 1e-12)

  peak_idx = int(np.argmax(power_db))
  est_elevation, est_azimuth = probe_points[peak_idx]

  print(f"True source direction : elevation={true_elevation:.1f} deg, azimuth={true_azimuth:.1f} deg")
  print(f"Estimated peak        : elevation={est_elevation:.1f} deg, azimuth={est_azimuth:.1f} deg ({power_db[peak_idx]:.2f} dB)")

  # Reshape into 2D map for visualisation
  heatmap_db = power_db.reshape(len(azimuths), len(elevations)).T
  print(f"2D acoustic map shape : {heatmap_db.shape}")


if __name__ == "__main__":
  main()
