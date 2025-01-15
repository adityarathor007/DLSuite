import unittest
import calc
from unittest.mock import patch
from employee import Employee

class TestCalc(unittest.TestCase):


#important to start with <test> in function name
    def test_add(self):
        self.assertEqual(calc.add(10,5),15)

    def test_subtract(self):
        self.assertEqual(calc.sub(10,4),6)

    def test_multiply(self):
        self.assertEqual(calc.multiply(10,8),80)
    
    def test_divide(self):
        self.assertEqual(calc.divide(10,5),2)
        with self.assertRaises(ValueError):
            calc.divide(10,0)
    
    #  class methods are methods that are bound to the class and not the instance of the class.
    @classmethod
    def setUpClass(cls):
        print('setupClass')
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')


    # run before each test method
    def setUp(self):
        print('setUp')
        self.emp1=Employee('aditya','rathor',1000000)
        self.emp2=Employee('G','sudheer',1000)

        

    # run after each test
    def tearDown(self):
        print('tearDown\n')
        


    def test_email(self):
        self.assertEqual(self.emp1.email,'aditya_rathor@gmail.com')
    
    def test_fullname(self):
        self.assertEqual(self.emp1.fullName,'aditya rathor')
    
    def test_apply_raise(self):
        self.assertEqual(self.emp1.pay*self.emp1.raise_amt,1050000)

    def test_monthly_schedule(self):
        # This allows control over the behavior of requests.get without actually making any real HTTP requests.

        with patch('employee.requests.get') as mocked_get:
            mocked_get.return_value.ok=True #intentionally setting the value to true to see what happens if value set to true
            mocked_get.return_value.text='Success'  #updating the return text should be this
            
            schedule=self.emp1.monthly_schedule('May')
            mocked_get.assert_called_with('http://company.com/rathor/May')  #checking if the request is being mdade to the correct url
            self.assertEqual(schedule,'Success')



            mocked_get.return_value.ok=False
            
            schedule=self.emp2.monthly_schedule('June')
            mocked_get.assert_called_with('http://company.com/sudheer/June')
            self.assertEqual(schedule,'Bad Response!')






if __name__=="__main__":
    unittest.main()