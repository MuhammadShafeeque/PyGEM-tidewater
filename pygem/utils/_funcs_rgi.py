"""
Python Glacier Evolution Model (PyGEM)

copyright © 2018 David Rounce <drounce@cmu.edu>

Distributed under the MIT license

Glacier ids and attribute tables of RGI 6.2 and RGI 7.0 (glaciers)
"""

import functools
import glob
import logging
import os

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

RGI_VERSIONS = ['62', '70G']
RGI7_PREFIX = 'RGI2000-v7.0-G-'

# RGI 7.0 attribute names and their RGI 6.0 names, as listed in the RGI 7.0 user guide
RGI7_TO_RGI6_COLUMNS = {
    'rgi_id': 'RGIId',
    'glims_id': 'GLIMSId',
    'cenlon': 'CenLon',
    'cenlat': 'CenLat',
    'area_km2': 'Area',
    'zmin_m': 'Zmin',
    'zmax_m': 'Zmax',
    'zmed_m': 'Zmed',
    'slope_deg': 'Slope',
    'aspect_deg': 'Aspect',
    'lmax_m': 'Lmax',
    'conn_lvl': 'Connect',
    'surge_type': 'Surging',
    'term_type': 'TermType',
    'glac_name': 'Name',
}
# RGI 7.0 terminus type 3 is shelf-terminating, which is 5 in RGI 6.0 (3 is dry calving there)
RGI7_TO_RGI6_TERMTYPE = {0: 0, 1: 1, 2: 2, 3: 5, 9: 9}


def check_rgi_version(rgi_version):
    if rgi_version not in RGI_VERSIONS:
        raise ValueError(f'rgi_version must be one of {RGI_VERSIONS}, not {rgi_version!r}')
    return rgi_version


def require_rgi6(rgi_version, tool):
    """Stop a tool whose inputs or file layout exist for RGI 6.2 only"""
    if check_rgi_version(rgi_version) != '62':
        raise NotImplementedError(f'{tool} supports rgi_version 62 (RGI 6.2) only, not {rgi_version}')


def is_rgi7_id(rgi_id):
    return str(rgi_id).startswith('RGI2000-')


def is_rgi7_glac_no(glac_no):
    """True for an RGI 7.0 glacier number ('05-09345'), False for RGI 6.0 ('5.13043')"""
    return '-' in str(glac_no)


def format_glac_no(glac_no, rgi_version):
    """Glacier number of the command line in the form selectglaciersrgitable takes: '05.13043' or '05-09345'"""
    if is_rgi7_glac_no(glac_no) != (check_rgi_version(rgi_version) == '70G'):
        raise ValueError(f'Glacier number {glac_no} does not belong to rgi_version {rgi_version}')
    if rgi_version == '70G':
        region, number = str(glac_no).removeprefix(RGI7_PREFIX).split('-')
        return f'{int(region):02d}-{number}'
    g = float(glac_no)
    return f'{g:.5f}' if g >= 10 else f'0{g:.5f}'


def split_glac_no(glac_no):
    """Region (int) and glacier number (str) of '5.13043' or '05-09345'"""
    region, number = str(glac_no).split('-' if is_rgi7_glac_no(glac_no) else '.')
    return int(region), number


def rgiid_from_glac_no(glac_no):
    """RGIId of a glacier number: '5.13043' -> 'RGI60-05.13043', '05-09345' -> 'RGI2000-v7.0-G-05-09345'"""
    region, number = split_glac_no(glac_no)
    if is_rgi7_glac_no(glac_no):
        return f'{RGI7_PREFIX}{region:02d}-{number}'
    return f'RGI60-{region:02d}.{number}'


def rgiid_region(rgi_id):
    """O1 region (int) of an RGI 6.0 or RGI 7.0 RGIId"""
    if is_rgi7_id(rgi_id):
        return int(rgi_id.split('-')[-2])
    return int(rgi_id.split('-')[1].split('.')[0])


def rgino_str(rgi_id):
    """Glacier number with the zero-padded region of an RGIId: '05.13043' or '05-09345'"""
    if is_rgi7_id(rgi_id):
        return rgi_id.removeprefix(RGI7_PREFIX)
    return rgi_id.split('-')[1]


def glacno(rgi_id):
    """Glacier number of an RGIId as in PyGEM's glacier table: '5.13043' or '05-09345'"""
    if is_rgi7_id(rgi_id):
        return rgino_str(rgi_id)
    region, number = split_glac_no(rgino_str(rgi_id))
    return f'{region}.{number}'


def prepro_rgi_version(rgi_id):
    """OGGM's prepro_rgi_version for an RGIId"""
    return '70G' if is_rgi7_id(rgi_id) else '62'


def glacier_str_from_table(glacier_rgi_table):
    """Glacier string of PyGEM's file names: '5.13043' (RGI 6.0) or '05-09345' (RGI 7.0)"""
    if is_rgi7_id(glacier_rgi_table['RGIId']):
        return glacier_rgi_table['glacno']
    return '{0:0.5f}'.format(glacier_rgi_table['RGIId_float'])


def region_str(glacier_str):
    """Two-digit region of a glacier string, as in PyGEM's output directories"""
    return str(split_glac_no(glacier_str)[0]).zfill(2)


def rgiid_region_dir(rgi_id):
    """Region directory of per-glacier files: 'RGI60-05' or 'RGI2000-v7.0-G-05'"""
    return rgi_id[:-6]


def rgi_region_csv(rgi_fp, region, rgi_version):
    """Glob pattern of the attribute table of one RGI region"""
    if check_rgi_version(rgi_version) == '70G':
        return f'{rgi_fp}/{RGI7_PREFIX}{str(region).zfill(2)}_*-attributes.csv'
    return f'{rgi_fp}/{str(region).zfill(2)}*.csv'


def rgi7_table_to_rgi6_columns(rgi7_table):
    """RGI 7.0 attribute table with the RGI 6.0 column names and codes the rest of PyGEM reads

    Columns without an RGI 6.0 name are kept under their RGI 7.0 name. EndDate is -9999 (RGI 7.0 has one date).
    """
    df = rgi7_table.rename(columns=RGI7_TO_RGI6_COLUMNS)
    df['O1Region'] = df['o1region'].astype(int)
    df['O2Region'] = [int(x.split('-')[1]) for x in df['o2region']]
    df['TermType'] = df['TermType'].map(RGI7_TO_RGI6_TERMTYPE)
    df['BgnDate'] = pd.to_datetime(df['src_date']).dt.strftime('%Y%m%d').astype(int)
    df['EndDate'] = -9999
    return df.drop(columns=['o1region', 'o2region'])


def read_rgi7_rgi6_links(links_fp, rgi7_id):
    """Rows of the RGI 7.0 to RGI 6.0 links table for one RGI 7.0 glacier

    links_fp is a links file or a directory holding the regional files (RGI2000-v7.0-G-<RR>_*-rgi6_links.csv).
    """
    if os.path.isdir(links_fp):
        fns = glob.glob(f'{links_fp}/{RGI7_PREFIX}{rgiid_region(rgi7_id):02d}_*rgi6_links.csv')
        if len(fns) != 1:
            raise FileNotFoundError(f'No single RGI 7.0 to RGI 6.0 links file for {rgi7_id} in {links_fp}')
        links_fp = fns[0]
    links = _read_links(links_fp)
    return links.loc[links['rgi7_id'] == rgi7_id]


@functools.lru_cache(maxsize=4)
def _read_links(links_fp):
    return pd.read_csv(links_fp)


def rgi6_links_fp(pygem_prms):
    """Path of the RGI 7.0 to RGI 6.0 links file or directory of the configuration, or None"""
    relpath = pygem_prms['rgi'].get('rgi7_rgi6_links_relpath')
    return f'{pygem_prms["root"]}/{relpath}' if relpath else None


def input_rgiids(rgi_id, links_fp):
    """Ids to look an input up under, in turn: the glacier's own, then for RGI 7.0 its one-to-one RGI 6.0 id"""
    yield rgi_id
    if is_rgi7_id(rgi_id):
        rgi6_id = rgi6_id_one_to_one(links_fp, rgi_id)
        if rgi6_id is None:
            log.warning(f'{rgi_id}: no one-to-one RGI 6.0 link, inputs held per RGI 6.0 id are not read')
        else:
            yield rgi6_id


def rgi6_id_one_to_one(links_fp, rgi7_id):
    """RGI 6.0 id linked one to one to an RGI 7.0 glacier (n_rgi6 = n_rgi7 = 1), otherwise None"""
    if links_fp is None:
        return None
    links = read_rgi7_rgi6_links(links_fp, rgi7_id)
    if len(links) == 1 and np.all(links[['n_rgi6', 'n_rgi7']].values == 1):
        return links['rgi6_id'].iloc[0]
    return None
