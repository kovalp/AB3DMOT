
from pathlib import Path

from ab3dmot_io.main import parse_args, main


def test_run_kitti(files_dir: Path) -> None:
    """Equivalent to run the script.

    `python3 main.py --dataset KITTI --det_name pointrcnn`
    """
    cli = parse_args(['--conf-root', str(files_dir / 'configs'),
                      '--data-root', str(files_dir / '../../data')])
    main(cli)
    


