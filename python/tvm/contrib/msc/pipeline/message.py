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
"""tvm.contrib.msc.pipeline.message"""

import datetime
import logging

from tvm.contrib.msc.core import utils as msc_utils
from tvm.contrib.msc.core.utils.namespace import MSCMap, MSCKey


def log_block(title: str, msg: str, logger: logging.Logger = None):
    """Log message in block format

    Parameters
    ----------
    title: str
        The title of the block
    msg: str
        The message to log.
    logger: logging.Logger
        The logger.
    """

    logger = logger or msc_utils.get_global_logger()
    if isinstance(msg, dict):
        msg = msc_utils.dump_dict(msg, "table")
    logger.info("\n{0} {1} {0}\n{2}\n{3} {1} {3}".format(">" * 20, title.center(40), msg, "<" * 20))


def time_stamp(mark: str, logger: logging.Logger = None):
    """Mark the stamp and record time.

    Parameters
    ----------
    mark: str
        The stamp name.
    logger: logging.Logger
        The logger.
    """

    logger = logger or msc_utils.get_global_logger()
    msg = "[MSC] Start {}".format(mark)
    logger.info("\n{0} {1} {0}\n".format("#" * 20, msg.center(40)))
    MSCMap.set(MSCKey.MSC_PHASE, mark)
    time_stamps = MSCMap.get(MSCKey.TIME_STAMPS, [])
    time_stamps.append((mark, datetime.datetime.now()))
    MSCMap.set(MSCKey.TIME_STAMPS, time_stamps)
