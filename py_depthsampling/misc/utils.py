"""Utilities."""

# Part of py_depthsampling library
# Copyright (C) 2017  Ingo Marquardt
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

from datetime import datetime
from dateutil.relativedelta import relativedelta


def year_diff(strDate01, strDate02):
    """Calculate difference in years between two dates.

    Parameters
    ----------
    strDate01 : str
        String representing the first date, format: YYYYMMDD, e.g. 20150930.
    strDate02 : str
        String representing the second date, format: YYYYMMDD, e.g. 20150930.

    Returns
    -------
    varYearDiff : int
        Difference in year between the two dates.
    """
    # Format first date string:
    objDte01 = datetime.strptime(strDate01, "%Y%m%d")

    # Format second date string:
    objDte02 = datetime.strptime(strDate02, "%Y%m%d")

    # Return difference of year between dates:
    return abs(relativedelta(objDte01, objDte02).years)
