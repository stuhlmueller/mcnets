## Copyright 2013 Google Inc. All Rights Reserved.
##
## Licensed under the Apache License, Version 2.0 (the )
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an AS IS BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import unittest

from mcnets import testutils


class TestStochasticTestCase(testutils.StochasticTestCase):

  def setUp(self):
    self.alpha = 0.05

  def testAssertSignificantlyGreater(self):
    a = [0, 0.1]
    b = [0.8, 0.9]
    self.assertNotSignificantlyGreater(a, b, alpha=self.alpha)
    self.assertSignificantlyGreater(b, a, alpha=self.alpha)
    a = [0, 0.5]
    b = [0.5, 1.0]
    self.assertNotSignificantlyGreater(a, b, alpha=self.alpha)
    self.assertNotSignificantlyGreater(b, a, alpha=self.alpha)
    a = [0.65, 0.82]
    b = [0.76, 0.85]
    self.assertNotSignificantlyGreater(a, b, alpha=self.alpha)
    self.assertNotSignificantlyGreater(b, a, alpha=self.alpha)
    a = [0.65, 0.82]
    b = [0.03, 0.04]
    self.assertSignificantlyGreater(a, b, alpha=self.alpha)
    self.assertNotSignificantlyGreater(b, a, alpha=self.alpha)


if __name__ == "__main__":
  unittest.main()
