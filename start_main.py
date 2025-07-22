"""."""

from sys import stderr

import numpy as np

from AB3DMOT_libs.model import AB3DMOT
from AB3DMOT_libs.utils import Config


def run() -> None:
    """."""
    cfg, settings_show = Config('configs/nuScenes.yml')
    print(cfg)
    tracker = AB3DMOT(cfg, 'Car', log=stderr)
    detections = {'dets': [np.ones(7, dtype='float32')], 'info': np.ones((1, 1))}
    results, affinity = tracker.track(detections, frame=0, seq_name='my-sequence')
    print(results)
    print(affinity)

    detections = {'dets': [      np.ones(7, dtype='float32'),
                           2.0 * np.ones(7, dtype='float32')], 'info': np.ones((2, 1))}
    tracker.track(detections, frame=1, seq_name='my-sequence')
    for track in tracker.trackers:
        print(track.id, track.hits)



if __name__ == '__main__':
    run()
