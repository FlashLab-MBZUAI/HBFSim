"""Native CPU DRAM and SSD isolation, shared dependencies, and exact receipts."""
import argparse
from pathlib import Path
import unittest

from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSession, SimulationSessionError
from hbfsim_client.transaction_protocol import Transaction, TransactionBatch


class HostDramAttachmentTests(unittest.TestCase):
    def session(self, dram=True, external=True):
        return SimulationSession(simulator_path=BINARY, system_config=SYSTEM,
            enable_hbm=True, enable_hbf=False, enable_external=external,
            host_dram_config=DRAM if dram else None)

    def test_independent_same_address_devices_join_before_hbm_write(self):
        with self.session() as session:
            result=session.submit(TransactionBatch(0, (
                Transaction('dram', 'HOST_DRAM', 'R', 0, 65536, 0),
                Transaction('ssd', 'EXTERNAL', 'R', 0, 4*2**20, 0),
                Transaction('fill', 'HBM', 'W', 0, 65536, 0, dependencies=('dram','ssd')),
            ), completions=True))
            by_id={r['id']:r for r in result['transaction_completions']}
            self.assertLess(by_id['dram']['finish_ns'],by_id['ssd']['finish_ns'])
            self.assertGreaterEqual(by_id['fill']['arrival_ns'],by_id['ssd']['finish_ns'])
            self.assertEqual(result['device_delta']['host_dram']['read_bytes'],65536)
            self.assertEqual(result['device_delta']['external']['read_bytes'],4*2**20)
            self.assertEqual(result['device_delta']['host_dram']['kind'],'host-dram')
            self.assertEqual(result['device_delta']['external']['kind'],'cxl-ssd')
            dram_finish=by_id['dram']['finish_ns']
        with self.session(external=False) as session:
            isolated=session.submit(TransactionBatch(0,(Transaction('dram','HOST_DRAM','R',0,65536,0),),completions=True))
            self.assertEqual(isolated['transaction_completions'][0]['finish_ns'],dram_finish)
            self.assertIsNone(isolated['device_delta']['external'])
            self.assertEqual(isolated['device_delta']['host_dram']['m2s_payload_bytes'],0)
            self.assertEqual(isolated['device_delta']['host_dram']['s2m_payload_bytes'],65536)

    def test_disabled_and_out_of_capacity_rejected(self):
        for dram,address in ((False,0),(True,DRAM.integer('external-backing-capacity-bytes'))):
            with self.session(dram=dram) as session:
                with self.assertRaises(SimulationSessionError):
                    session.submit(TransactionBatch(0,(Transaction('bad','HOST_DRAM','W',address,4096,0),)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--simulator',type=Path,required=True)
    args=parser.parse_args()
    BINARY=args.simulator.resolve();ROOT=Path(__file__).resolve().parents[2]
    SYSTEM=ResolvedSystemConfig.load([ROOT/'configs/systems/eight-stack-baseline.cfg',ROOT/'configs/overlays/backing/cxl-ssd.cfg',ROOT/'configs/overlays/backing/cxl-ssd-cached.cfg']).resolve(BINARY,enable_hbf=False)
    DRAM=ResolvedSystemConfig.load([ROOT/'configs/systems/eight-stack-baseline.cfg',ROOT/'configs/overlays/backing/host-dram.cfg']).resolve(BINARY,enable_hbf=False)
    unittest.main(argv=['test_host_dram_attachment'])
