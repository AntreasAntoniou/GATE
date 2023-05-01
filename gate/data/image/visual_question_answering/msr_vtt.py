import os
import json
import pathlib
from opendatalab.__version__ import __url__
from opendatalab.cli.get import implement_get
from opendatalab.cli.info import implement_info
from opendatalab.cli.login import implement_login
from opendatalab.cli.ls import implement_ls
from opendatalab.cli.search import implement_search
from opendatalab.cli.utility import ContextInfo
from unittest.mock import patch
import multiprocessing as mp

if __name__ == "__main__":
    ctx = ContextInfo(__url__, "")
    client = ctx.get_client()
    odl_api = client.get_api()
    implement_login(
        obj=ctx,
        username=os.environ.get("ODL_USERNAME"),
        password=os.environ.get("ODL_PASSWORD"),
    )

    def always_true(*args, **kwargs):
        return True

    with patch("click.confirm", side_effect=always_true):
        implement_get(
            ctx,
            "MSR-VTT",
            pathlib.Path(os.environ.get("PYTEST_DIR")) / "msr_vtt",
            min(mp.cpu_count(), 8),
        )
