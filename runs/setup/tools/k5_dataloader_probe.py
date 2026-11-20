"""CPU-only real tensor IPC regression; private scratch, never a science launch."""
import json
import multiprocessing
import multiprocessing.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k0_isolated_release as adapter


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    identity = 'k5-cpu-probe-' + uuid.uuid4().hex
    short = None
    try:
        with tempfile.TemporaryDirectory(prefix='k5-bootstrap-') as scratch:
            root = Path(scratch)
            code = Path(__file__).resolve().parents[3]
            row = {'script': 'scripts/dsreset_train.py', 'args': []}
            grant = {'rows': {identity: {'row': row}}, 'kind': 'synthetic-cpu-only',
                     'adapter_sha256': 'test', 'worker_sha256': 'test'}
            with patch.object(adapter, 'MAIN', root), patch.object(adapter, 'IDS', {identity}), patch.object(
                    adapter, 'load_authorization', return_value=(grant, {'code_root': str(code)}, 'test')):
                adapter.prepare_worker(root / 'not-a-live-grant', identity, [])
                old = os.environ['TMPDIR']
                assert len(os.fsencode(old + '/pymp-12345678/listener-12345678')) > 107
                tempfile.tempdir = old
                multiprocessing.current_process()._config['tempdir'] = old
                short = adapter.configure_worker_temp(identity)
                assert tempfile.gettempdir() == str(short)
                assert multiprocessing.util.get_temp_dir().startswith(str(short) + '/')
                for name in ('TMPDIR', 'TEMP', 'TMP', 'XDG_RUNTIME_DIR'):
                    assert os.environ[name] == str(short)
                import torch
                from torch.utils.data import DataLoader, Dataset
                torch.set_num_threads(4)
                torch.set_num_interop_threads(1)
                torch.multiprocessing.set_sharing_strategy('file_descriptor')
                class Samples(Dataset):
                    def __len__(self):
                        return 96
                    def __getitem__(self, index):
                        return torch.tensor([index, os.getpid()], dtype=torch.int64)
                loader = DataLoader(Samples(), batch_size=4, num_workers=6,
                                    prefetch_factor=2, persistent_workers=True, pin_memory=True,
                                    multiprocessing_context='fork', timeout=20)
                pids = set()
                for epoch in range(2):
                    indices = []
                    for batch in loader:
                        indices.extend(batch[:, 0].tolist())
                        pids.update(batch[:, 1].tolist())
                    assert indices == list(range(96))
                sockets = [p for p in short.glob('pymp-*/listener-*') if p.is_socket()]
                assert len(pids) == 6 and len(sockets) >= 6
                assert all(len(os.fsencode(p)) <= 107 for p in sockets)
                print(json.dumps({'samples': 192, 'epochs': 2, 'worker_pids': sorted(pids),
                                  'workers': 6, 'prefetch_factor': 2, 'persistent_workers': True,
                                  'pin_memory_requested': True, 'device': 'cpu',
                                  'listener_sockets': len(sockets),
                                  'max_listener_path_bytes': max(len(os.fsencode(p)) for p in sockets),
                                  'bootstrap_tmp_bytes': len(os.fsencode(old)), 'tmp': str(short)}))
                loader._iterator._shutdown_workers()
    finally:
        if short is not None:
            shutil.rmtree(short)  # only this UUID test identity; never a campaign row


if __name__ == '__main__':
    main()
