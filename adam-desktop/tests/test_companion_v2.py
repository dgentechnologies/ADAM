import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cloud_sync import CloudSync, CloudSyncError, validate_companion, merge_companions

class Account:
    user = {'uid': 'one'}

class SharedRecordsTest(unittest.TestCase):
    def test_multiple_devices_planner_ble_and_account_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            account = Account()
            sync = CloudSync(account, Path(directory))
            first = sync.save_record('devices', {'name': 'Desk', 'serial': 'SIM-1', 'simulated': True})
            second = sync.save_record('devices', {'name': 'Studio', 'serial': 'SIM-2', 'simulated': True})
            todo = sync.save_record('todos', {'text': 'Build ADAM', 'done': False, 'dueAt': '', 'deviceId': first['id']})
            clock = sync.save_record('clocks', {'kind': 'timer', 'label': 'Focus', 'when': '2026-10-10T12:00:00.000Z', 'enabled': True, 'deviceId': first['id']})
            snapshot = sync.sync_simulated_ble(first['id'])['companion']
            self.assertEqual(len(snapshot['devices']), 2)
            self.assertIn(todo['id'], snapshot['todos'])
            self.assertIn(clock['id'], snapshot['clocks'])
            sync.delete_record('todos', todo['id'])
            sync.save_record('devices', {**second, 'name': 'Bedroom'})
            self.assertTrue(sync.sync_simulated_ble(first['id'])['companion']['todos'][todo['id']]['deleted'])
            account.user = {'uid': 'two'}
            self.assertEqual(sync.list_records('devices'), [])
            with self.assertRaises(CloudSyncError): sync.sync_simulated_ble(first['id'])
            account.user = {'uid': 'one'}
            restored = CloudSync(account, Path(directory))
            self.assertEqual(restored.list_records('todos'), [])
            self.assertEqual([d['name'] for d in restored.list_records('devices') if d['id'] == second['id']], ['Bedroom'])

    def test_common_fixture_and_invalid_records(self):
        fixture = json.loads((Path(__file__).resolve().parents[2] / 'shared/companion-v2.fixture.json').read_text())
        self.assertEqual(validate_companion(fixture), fixture)
        self.assertEqual(merge_companions(fixture, fixture), fixture)
        for field in ('todos', 'clocks', 'devices'):
            with self.assertRaises(CloudSyncError): validate_companion({**fixture, field: []})

if __name__ == '__main__': unittest.main()
