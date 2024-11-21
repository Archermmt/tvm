# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Test trace in MSC. """

import tvm
from tvm.contrib.msc.core.frontend import msc_trace, dump_traced
from tvm.contrib.msc.core import utils as msc_utils


def test_binary_ops():
    """Test tracker for binary ops"""

    @msc_trace()
    def numpy_func(data):
        data = data + 1
        data = data - 2
        data = data * data
        data /= 5
        data *= 2
        return data

    # input_info=[([256, 256], "float32")]
    data = msc_utils.random_data([(256, 256), "float32"])
    res = numpy_func(data)
    print("res " + str(msc_utils.inspect_array(res)))
    configs = dump_traced()
    print("configs " + str(configs))


if __name__ == "__main__":
    # tvm.testing.main()
    test_binary_ops()
