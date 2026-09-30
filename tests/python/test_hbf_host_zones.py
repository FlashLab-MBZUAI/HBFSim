#!/usr/bin/env python3
"""End-to-end host zone restart and default wear artifact contracts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession
SIMULATOR = ROOT/'build/hbfsim'


class HostZoneTests(unittest.TestCase):
    def test_fragment_writes_reach_nand_once_and_timeout_is_retryable(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            override = out/'fragments.cfg'
            override.write_text('hbf-write-accumulation-timeout-ns=10000\n'
                                'hbf-outstanding-write-pages-per-channel=1\n')
            config = ResolvedSystemConfig.load((ROOT/'configs/systems/eight-stack-baseline.cfg',
                                                ROOT/'tests/fixtures/hbf-host-zones.cfg', override))
            with SimulationSession(simulator_path=SIMULATOR, system_config=config,
                    enable_hbm=True, enable_hbf=True, hbf_wear_output_prefix=out/'wear') as session:
                fragments = [dict(id=f'line-{i}', arrival_offset_ns=i*10,
                                  address=i*64, bytes=64) for i in range(64)]
                assembled = session.hbf_channel_write_batch('assembled', fragments)
                commands = assembled['commands']
                self.assertEqual([c['status'] for c in commands], [0]*64)
                self.assertEqual(len({c['finish_ns'] for c in commands}), 1)
                self.assertEqual(sum(c['physical_bytes'] for c in commands), 4096)
                self.assertEqual(sum(c['array_program_ns'] > 0 for c in commands), 1)
                self.assertGreater(assembled['finish_ns'], max(c['received_ns'] for c in commands))
                session.hbf_zone_command('READ', 'assembled-read', zone=0, argument=64)
                failed = session.hbf_channel_write_batch('missing', [
                    dict(id='partial', arrival_offset_ns=0, address=4096, bytes=64),
                    dict(id='overlap', arrival_offset_ns=10, address=4096, bytes=128)])
                self.assertEqual([c['status'] for c in failed['commands']], [5, 2])
                self.assertEqual(sum(c['physical_bytes'] for c in failed['commands']), 0)
                partial = failed['commands'][0]
                self.assertAlmostEqual(partial['finish_ns'] - partial['received_ns'], 10000)
                retry = session.hbf_channel_write_batch('retry', [
                    dict(id='retry-page', arrival_offset_ns=0, address=4096, bytes=4096)])
                self.assertEqual(retry['commands'][0]['status'], 0)
                self.assertEqual(retry['commands'][0]['physical_bytes'], 4096)
                session.hbf_zone_command('READ', 'retry-read', zone=4096, argument=64)
            wear = json.loads(Path(session.hbf_wear_artifacts['json']).read_text())
            self.assertEqual(sum(b['valid_pages'] for b in wear['bins']), 2)
            self.assertEqual(sum(b['workload_erases'] for b in wear['bins']), 1)

    def test_compact_population_and_explicit_recycle(self):
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            raw=out/'raw.cfg'
            raw.write_text('hbf-mapping-mode=raw-physical\nhbf-write-buffer-pages=0\n'
                           'hbf-write-buffer-flush-threshold-pages=0\n')
            config=ResolvedSystemConfig.load((ROOT/'configs/systems/eight-stack-baseline.cfg',
                                              ROOT/'tests/fixtures/hbf-host-zones.cfg',raw))
            zone_bytes=4*8*4096
            with SimulationSession(simulator_path=SIMULATOR,system_config=config,
                    enable_hbm=True,enable_hbf=True,hbf_wear_output_prefix=out/'wear') as session:
                session.hbf_zone_command('POPULATE','initial',zone=0,argument=2)
                session.hbf_zone_command('READ','initial-data',zone=0,argument=4096)
                for i in range(2):
                    session.hbf_zone_command('INVALIDATE',f'expire-{i}',zone=0)
                    session.hbf_zone_command('RECYCLE',f'recycle-{i}',zone=0)
                    session.hbf_zone_command('WRITE',f'rewrite-{i}',zone=0,argument=zone_bytes)
                session.hbf_zone_command('READ','live-neighbor',zone=zone_bytes,argument=4096)
                session.checkpoint_image('save',out/'state.image')
            self.assertEqual((out/'state.image').read_text().splitlines()[0],
                             'HBFSIM_HBF_PERSISTENT_IMAGE 7')
            wear=json.loads(Path(session.hbf_wear_artifacts['json']).read_text())
            self.assertEqual(sum(b['workload_erases'] for b in wear['bins']),8)
            self.assertEqual(wear['host_zone_remaps'],1)
            self.assertEqual(wear['host_zone_resets'],2)
            self.assertEqual(wear['gc_relocations'],0)

    def test_clean_input_eof_emits_final_wear_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)/'eof-wear'
            result = subprocess.run([str(SIMULATOR), '--system-config',
                str(ROOT/'configs/systems/eight-stack-baseline.cfg'), '--system-config',
                str(ROOT/'tests/fixtures/hbf-host-zones.cfg'), '--enable-hbm', 'true',
                '--hbf-wear-output-prefix', str(prefix)],
                input='ZONE_WRITE data 0 0 0 4096\n', text=True, capture_output=True, check=True)
            stop = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(stop['result'], 'stopped')
            self.assertTrue(Path(stop['hbf_wear_artifacts']['html']).is_file())
            snapshot = json.loads(Path(stop['hbf_wear_artifacts']['json']).read_text())
            self.assertEqual(sum(b['valid_pages'] for b in snapshot['bins']), 1)

    def test_restart_preserves_mapping_and_separates_workload_wear(self):
        config = ResolvedSystemConfig.load((ROOT/'configs/systems/eight-stack-baseline.cfg',
                                           ROOT/'tests/fixtures/hbf-host-zones.cfg'))
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with SimulationSession(simulator_path=SIMULATOR,system_config=config,
                    enable_hbm=True,enable_hbf=True,hbf_wear_output_prefix=out/'before') as first:
                for i in range(12):
                    first.hbf_zone_command('WRITE',f'w-{i}',zone=0,argument=4096)
                    first.hbf_zone_command('INVALIDATE',f'i-{i}',zone=0)
                    first.hbf_zone_command('REMAP',f'swap-{i}',zone=0,argument=1)
                    first.hbf_zone_command('RESET',f'e-{i}',zone=0)
                first.hbf_zone_command('WRITE','final-data',zone=0,argument=4096)
                first.checkpoint_image('image',out/'media.image')
            before = json.loads(Path(first.hbf_wear_artifacts['json']).read_text())
            self.assertEqual(sum(b['workload_erases'] for b in before['bins']),13)
            self.assertGreater(before['host_zone_remaps'],0)
            self.assertEqual(before['host_gc_read_bytes'],0)
            self.assertEqual(before['gc_relocations'],0)
            self.assertTrue(Path(first.hbf_wear_artifacts['html']).is_file())
            with SimulationSession(simulator_path=SIMULATOR,system_config=config,
                    enable_hbm=True,enable_hbf=True,initial_hbf_persistent_image=out/'media.image',
                    hbf_wear_output_prefix=out/'after') as second:
                second.hbf_zone_command('READ','restored-data',zone=0,argument=4096)
            after=json.loads(Path(second.hbf_wear_artifacts['json']).read_text())
            self.assertEqual(after['zone_remapping'],before['zone_remapping'])
            self.assertEqual([b['pec_sum'] for b in after['bins']],
                             [b['pec_sum'] for b in before['bins']])
            self.assertEqual(sum(b['workload_erases'] for b in after['bins']),0)
            self.assertEqual(sum(b['initial_pec_sum'] for b in after['bins']),13)
            for b in after['bins']:
                self.assertEqual(sum(b[k] for k in ('valid_pages','invalid_pages','free_pages','pending_pages')),
                                 (b['block_end']-b['block_begin'])*8)

    def test_page_zero_erase_report_requires_no_traffic_collector(self):
        config=ResolvedSystemConfig.load((ROOT/'configs/systems/eight-stack-baseline.cfg',
                                         ROOT/'tests/fixtures/hbf-host-zones.cfg'))
        with tempfile.TemporaryDirectory() as temporary:
            previous=Path.cwd()
            try:
                os.chdir(temporary)
                with SimulationSession(simulator_path=SIMULATOR,system_config=config,
                        enable_hbm=True,enable_hbf=True) as session:
                    session.hbf_zone_command('WRITE','no-erase',zone=0,argument=4096)
                artifacts=session.hbf_wear_artifacts
                self.assertTrue(Path(artifacts['html']).is_file())
                snapshot=json.loads(Path(artifacts['json']).read_text())
                self.assertEqual(sum(b['workload_erases'] for b in snapshot['bins']),1)
                self.assertEqual(snapshot['blocks'],128)
                self.assertEqual(snapshot['host_gc_read_bytes'],0)
            finally:
                os.chdir(previous)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--simulator',type=Path,default=SIMULATOR)
    options,remaining=parser.parse_known_args()
    SIMULATOR=options.simulator.resolve()
    unittest.main(argv=[sys.argv[0],*remaining])
