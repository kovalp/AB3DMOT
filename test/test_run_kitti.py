
from ab3dmot_io.main import parse_args, main


def test_run_kitti() -> None:
    """Equivalent to run the script.

    `python3 main.py --dataset KITTI --det_name pointrcnn`
    """
    cli = parse_args([])
    main(cli)
    


