import logging

import numpy as np
import pandas as pd
import pytest
import ruamel.yaml

import pygem.pygem_modelsetup as modelsetup
from pygem.setup.config import ConfigManager
from pygem.utils import _funcs_rgi as rgi

"""
Test suite for the RGI 6.2 and RGI 7.0 glacier ids, attribute tables and links table"""

RGI7_ID = 'RGI2000-v7.0-G-05-09345'
RGI6_ID = 'RGI60-05.13043'

RGI7_COLUMNS = (
    'rgi_id,o1region,o2region,glims_id,anlys_id,subm_id,src_date,cenlon,cenlat,utm_zone,area_km2,primeclass,'
    'conn_lvl,surge_type,term_type,glac_name,is_rgi6,termlon,termlat,zmin_m,zmax_m,zmed_m,zmean_m,slope_deg,'
    'aspect_deg,aspect_sec,dem_source,lmax_m'
)
RGI7_ROWS = [
    f'{RGI7_ID},05,05-01,G330486E68306N,795933,751,1999-08-11T00:00:00,-29.518,68.305,26,21.087,0,0,0,1,,1,'
    '-29.513,68.275,-2.65,824.6,386.6,368.3,10.6,276.3,7.0,COPDEM90,8667',
    'RGI2000-v7.0-G-05-09346,05,05-02,G330000E68000N,1,1,2001-07-01T00:00:00,-30.0,68.0,26,1.5,0,1,0,9,,0,'
    '-30.0,68.0,100,900,500,500,20,90,3.0,COPDEM90,1000',
    'RGI2000-v7.0-G-05-09347,05,05-01,G331000E68000N,1,1,2001-07-01T00:00:00,-29.0,68.0,26,2.5,0,0,0,3,,0,'
    '-29.0,68.0,0,800,400,400,15,180,5.0,COPDEM90,2000',
]
LINKS = [
    'rgi7_id,rgi6_id,rgi7_reg,rgi6_reg,overlap_area_km2,rgi7_area_fraction,rgi6_area_fraction,cluster_id,n_rgi7,n_rgi6',
    f'{RGI7_ID},{RGI6_ID},05,05,21.0866,0.99998,0.99998,1,1,1',
    'RGI2000-v7.0-G-05-09346,RGI60-05.13044,05,05,0.7,0.47,0.5,2,2,1',
    'RGI2000-v7.0-G-05-09347,RGI60-05.13044,05,05,0.7,0.28,0.5,2,2,1',
]


@pytest.fixture
def rgi7_dir(tmp_path):
    (tmp_path / 'RGI2000-v7.0-G-05_greenland_periphery-attributes.csv').write_text(
        '\n'.join([RGI7_COLUMNS] + RGI7_ROWS) + '\n'
    )
    (tmp_path / 'RGI2000-v7.0-G-05_greenland_periphery-rgi6_links.csv').write_text('\n'.join(LINKS) + '\n')
    return tmp_path


@pytest.mark.parametrize('glac_no', ['1.03622', '5.13043', '13.00001', 5.13043, '5.1304'])
def test_format_glac_no_rgi6_unchanged(glac_no):
    g = float(glac_no)
    assert rgi.format_glac_no(glac_no, '62') == (f'{g:.5f}' if g >= 10 else f'0{g:.5f}')


@pytest.mark.parametrize('glac_no', ['05-09345', '5-09345', RGI7_ID])
def test_format_glac_no_rgi7(glac_no):
    assert rgi.format_glac_no(glac_no, '70G') == '05-09345'


@pytest.mark.parametrize('glac_no, rgi_version', [('05-09345', '62'), ('5.13043', '70G'), ('5.13043', '70')])
def test_format_glac_no_wrong_version(glac_no, rgi_version):
    with pytest.raises(ValueError):
        rgi.format_glac_no(glac_no, rgi_version)


def test_ids_rgi6():
    rgiid = 'RGI60-01.03622'
    assert rgi.rgiid_from_glac_no('1.03622') == rgiid
    assert rgi.rgiid_from_glac_no('01.03622') == rgiid
    assert rgi.rgiid_region(rgiid) == 1
    assert rgi.rgino_str(rgiid) == '01.03622'
    assert rgi.glacno(rgiid) == '1.03622'
    assert rgi.region_str('1.03622') == '01'
    assert rgi.rgiid_region_dir(rgiid) == 'RGI60-01'
    assert rgi.prepro_rgi_version(rgiid) == '62'
    row = pd.Series({'RGIId': rgiid, 'RGIId_float': 1.03622, 'glacno': '1.03622'})
    assert rgi.glacier_str_from_table(row) == '{0:0.5f}'.format(1.03622)


def test_ids_rgi7():
    assert rgi.rgiid_from_glac_no('05-09345') == RGI7_ID
    assert rgi.rgiid_region(RGI7_ID) == 5
    assert rgi.rgino_str(RGI7_ID) == '05-09345'
    assert rgi.glacno(RGI7_ID) == '05-09345'
    assert rgi.region_str('05-09345') == '05'
    assert rgi.rgiid_region_dir(RGI7_ID) == 'RGI2000-v7.0-G-05'
    assert rgi.prepro_rgi_version(RGI7_ID) == '70G'
    row = pd.Series({'RGIId': RGI7_ID, 'RGIId_float': np.nan, 'glacno': '05-09345'})
    assert rgi.glacier_str_from_table(row) == '05-09345'


def test_rgi7_table_to_rgi6_columns(rgi7_dir):
    df = rgi.rgi7_table_to_rgi6_columns(pd.read_csv(next(rgi7_dir.glob('*-attributes.csv'))))
    assert list(df['RGIId']) == [RGI7_ID, 'RGI2000-v7.0-G-05-09346', 'RGI2000-v7.0-G-05-09347']
    assert list(df['O1Region']) == [5, 5, 5]
    assert list(df['O2Region']) == [1, 2, 1]
    # shelf-terminating is 3 in RGI 7.0 and 5 in RGI 6.0
    assert list(df['TermType']) == [1, 9, 5]
    assert list(df['BgnDate']) == [19990811, 20010701, 20010701]
    assert list(df['EndDate']) == [-9999] * 3
    assert df['Area'].iloc[0] == 21.087


def test_selectglaciersrgitable_rgi7(rgi7_dir):
    kwargs = dict(rgi_fp=str(rgi7_dir), rgi_version='70G', glac_no_skip=None)
    table = modelsetup.selectglaciersrgitable(glac_no=['05-09345'], **kwargs)
    assert len(table) == 1
    row = table.loc[0]
    assert row['RGIId'] == RGI7_ID
    assert row['glacno'] == row['rgino_str'] == '05-09345'
    assert np.isnan(row['RGIId_float'])
    assert (row['O1Region'], row['O2Region'], row['TermType'], row['Area']) == (5, 1, 1, 21.087)
    assert row['CenLon_360'] == pytest.approx(360 - 29.518)
    # the whole region: Connect 1 is kept, as in RGI 6.0
    table = modelsetup.selectglaciersrgitable(rgi_regionsO1=[5], **kwargs)
    assert list(table['glacno']) == ['05-09345', '05-09346', '05-09347']
    table = modelsetup.selectglaciersrgitable(rgi_regionsO1=[5], include_landterm=False, **kwargs)
    assert list(table['glacno']) == ['05-09345', '05-09347']


def test_selectglaciersrgitable_rgi7_wrong_version(rgi7_dir):
    with pytest.raises(ValueError):
        modelsetup.selectglaciersrgitable(glac_no=['05-09345'], rgi_fp=str(rgi7_dir), rgi_version='7')


def test_links_one_to_one(rgi7_dir):
    links_fp = next(rgi7_dir.glob('*-rgi6_links.csv'))
    for fp in [str(links_fp), str(rgi7_dir)]:
        assert rgi.rgi6_id_one_to_one(fp, RGI7_ID) == RGI6_ID
        # an RGI 6.0 glacier split into two RGI 7.0 glaciers is not one to one
        assert rgi.rgi6_id_one_to_one(fp, 'RGI2000-v7.0-G-05-09346') is None
        assert rgi.rgi6_id_one_to_one(fp, 'RGI2000-v7.0-G-05-00001') is None
    assert rgi.rgi6_id_one_to_one(None, RGI7_ID) is None


def test_input_rgiids(rgi7_dir, caplog):
    assert list(rgi.input_rgiids(RGI6_ID, None)) == [RGI6_ID]
    assert list(rgi.input_rgiids(RGI7_ID, str(rgi7_dir))) == [RGI7_ID, RGI6_ID]
    # the links table is only read when the glacier's own id is not enough
    ids = rgi.input_rgiids(RGI7_ID, str(rgi7_dir / 'missing'))
    assert next(ids) == RGI7_ID
    with caplog.at_level(logging.WARNING):
        assert list(rgi.input_rgiids('RGI2000-v7.0-G-05-09346', str(rgi7_dir))) == ['RGI2000-v7.0-G-05-09346']
    assert 'no one-to-one RGI 6.0 link' in caplog.text


def test_require_rgi6():
    rgi.require_rgi6('62', 'tool')
    with pytest.raises(NotImplementedError, match='tool supports rgi_version 62'):
        rgi.require_rgi6('70G', 'tool')


def test_config_without_rgi_keys(tmp_path):
    """A configuration file written before rgi_version existed reads with the RGI 6.2 defaults"""
    cm = ConfigManager(base_dir=str(tmp_path), check_paths=False)
    ryaml = ruamel.yaml.YAML()
    with open(cm.config_path) as f:
        config = ryaml.load(f)
    assert config['rgi']['rgi_version'] == '62'
    del config['rgi']['rgi_version']
    del config['rgi']['rgi7_rgi6_links_relpath']
    with open(cm.config_path, 'w') as f:
        ryaml.dump(config, f)
    config = cm.read_config()
    assert config['rgi']['rgi_version'] == '62'
    assert config['rgi']['rgi7_rgi6_links_relpath'] is None
    cm.update_config({'rgi.rgi_version': '70G'})
    assert cm.read_config()['rgi']['rgi_version'] == '70G'
