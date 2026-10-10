import glob
import os
import subprocess
import sys

import pytest
import xarray as xr
from oggm.core.flowline import k_calving_law

from pygem.setup.config import ConfigManager

"""
Test suite to any necessary aux"""


@pytest.fixture(scope='module')
def rootdir():
    config_manager = ConfigManager()
    pygem_prms = config_manager.read_config()
    return pygem_prms['root']


def test_simulation_massredistribution_dynamics(rootdir):
    """
    Test the run_simulation CLI script with the "MassRedistributionCurves" dynamical option.
    """

    # Run run_simulation CLI script
    subprocess.run(
        [
            'run_simulation',
            '-rgi_glac_number',
            '1.03622',
            '-option_calibration',
            'MCMC',
            '-sim_climate_name',
            'ERA5',
            '-sim_startyear',
            '2000',
            '-sim_endyear',
            '2019',
            '-nsims',
            '1',
            '-option_dynamics',
            'MassRedistributionCurves',
            '-outputfn_sfix',
            'mrcdynamics_',
        ],
        check=True,
    )

    # Check if output files were created
    outdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'ERA5')
    output_files = glob.glob(os.path.join(outdir, '**', '*_mrcdynamics_all.nc'), recursive=True)
    assert output_files, f'Simulation output file not found in {outdir}'


def test_simulation_no_dynamics(rootdir):
    """
    Test the run_simulation CLI script with no dynamics option.
    """

    # Run run_simulation CLI script
    subprocess.run(
        [
            'run_simulation',
            '-rgi_glac_number',
            '1.03622',
            '-option_calibration',
            'MCMC',
            '-sim_climate_name',
            'ERA5',
            '-sim_startyear',
            '2000',
            '-sim_endyear',
            '2019',
            '-nsims',
            '1',
            '-option_dynamics',
            'None',
            '-outputfn_sfix',
            'nodynamics_',
        ],
        check=True,
    )

    # Check if output files were created
    outdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'ERA5')
    output_files = glob.glob(os.path.join(outdir, '**', '*_nodynamics_all.nc'), recursive=True)
    assert output_files, f'Simulation output file not found in {outdir}'


def k_calving_law_annual(model, flowline, last_above_wl, year):
    """
    The k-calving law of the "MassRedistributionCurves" dynamical option, as a calving law: frontal ablation of the
    year (m3 ice).
    """
    h = flowline.thick[last_above_wl]
    d = h - (flowline.surface_h[last_above_wl] - model.water_level)
    return model.calving_k * d * h * flowline.widths_m[last_above_wl]


@pytest.mark.parametrize(
    'option_dynamics, calving_law',
    [('OGGM', k_calving_law), ('MassRedistributionCurves', k_calving_law_annual)],
)
def test_simulation_calving_law_hook(rootdir, monkeypatch, option_dynamics, calving_law):
    """
    Test the calving law hook of run_simulation: the hook is called for the tidewater glacier with the dynamical
    option, and a hook returning the default k-calving law reproduces the simulation without a hook.
    """
    from pygem.bin.run import run_simulation

    assert run_simulation.calving_law_for is None

    calls = []

    def calving_law_for(gdir, option_dynamics):
        calls.append((gdir.rgi_id, option_dynamics))
        return calving_law

    outdir = os.path.join(rootdir, 'Output', 'simulations', '01', 'ERA5')
    datasets = {}
    for sfix, hook in [('nohook_', None), ('hook_', calving_law_for)]:
        monkeypatch.setattr(run_simulation, 'calving_law_for', hook)
        monkeypatch.setattr(
            sys,
            'argv',
            [
                'run_simulation',
                '-rgi_glac_number',
                '1.03622',
                '-option_calibration',
                'MCMC',
                '-sim_climate_name',
                'ERA5',
                '-sim_startyear',
                '2000',
                '-sim_endyear',
                '2019',
                '-nsims',
                '1',
                '-option_dynamics',
                option_dynamics,
                '-outputfn_sfix',
                f'calvinglaw_{option_dynamics}_{sfix}',
            ],
        )
        run_simulation.main()

        output_files = glob.glob(
            os.path.join(outdir, '**', f'*_calvinglaw_{option_dynamics}_{sfix}all.nc'), recursive=True
        )
        assert output_files, f'Simulation output file not found in {outdir}'
        datasets[sfix] = xr.open_dataset(output_files[0]).load()

    assert calls == [('RGI60-01.03622', option_dynamics)]
    xr.testing.assert_equal(datasets['hook_'], datasets['nohook_'])
