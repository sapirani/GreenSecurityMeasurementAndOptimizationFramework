import unittest
from unittest.mock import patch, MagicMock
from splunk_tools import SplunkTools
from dotenv import load_dotenv

class TestSplunkTools(unittest.TestCase):

    @patch('requests.Session')
    def setUp(self, mock_session):
        self.splunk_tools = SplunkTools()


    @patch('requests.post')
    def test_insert_log(self, mock_post):
        log_entry = "Test log entry"
        log_source = "Test log source"

        mock_post.return_value.status_code = 200
        self.splunk_tools.insert_log(log_entry, log_source)
        mock_post.assert_called_once()

    @patch('subprocess.run')
    def test_extract_distribution(self, mock_run):
        start_time = "2023-07-23 00:00:00"
        end_time = "2023-07-23 23:59:59"

        mock_run.return_value.stdout = "Test output"
        self.splunk_tools.extract_distribution(start_time, end_time)
        mock_run.assert_called_once()

    @patch('requests.Session.get')
    @patch('json.loads')
    def test_get_search_details(self, mock_json_loads, mock_get):
        search = "Test search"
        res_dict = {"Test search": ["Test rule", "Test time", "Test total events", "Test total run time"]}
        search_endpoint = "Test search endpoint"

        mock_get.return_value.text = "Test text"
        mock_json_loads.return_value = {"entry": [{"content": {"pid": 1, "runDuration": "Test run duration"}}]}
        self.splunk_tools.get_search_details(search, res_dict, search_endpoint)
        mock_get.assert_called_once()
        mock_json_loads.assert_called_once()

    # @patch('subprocess.run')
    # @patch('splunk_tools.SplunkTools.get_search_details')
    # def test_get_searches_and_jobs_info(self, mock_get_search_details, mock_run):
    #     time = "60"

    #     mock_run.return_value.stdout = "Test output\nTest output\n"
    #     mock_get_search_details.return_value = ("Test rule", ("Test search id", 1, "Test time", "Test run duration", "Test total events", "Test total run time"))
    #     self.splunk_tools.get_searches_and_jobs_info(time)
    #     mock_run.assert_called_once()
    #     mock_get_search_details.assert_called_once()

    @patch('requests.Session.post')
    @patch('json.loads')
    def test_extract_alerts(self, mock_json_loads, mock_post):
        mock_post.return_value.text = "Test text"
        mock_json_loads.return_value = {"Test key": "Test value"}
        self.assertEqual(self.splunk_tools.extract_alerts(), {"Test key": "Test value"})
        mock_post.assert_called_once()
        mock_json_loads.assert_called_once()

if __name__ == '__main__':
    unittest.main()
