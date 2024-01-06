import unittest
from fastapi.testclient import TestClient
from fastapi import status
from main import app


client = TestClient(app)


class TestStringMethods(unittest.TestCase):

    def test_client(self):
        response = client.get("/infer")
        json_response = response.json()
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        output_name = json_response['outputs'][0]['name']
        self.assertEqual(output_name, '1277')


if __name__ == '__main__':
    unittest.main()