import inspect
import os
from importlib.resources import files
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from tabsim import config
from tabsim.config import (
    add_astro_sources,
    deep_update,
    get_telescope_definitions,
    load_config,
    load_obs,
    yaml_load,
)
from tabsim.sky import generate_random_sky, random_power_law

# The settings of `ast_sources.<type>.random` and the arguments that they set
SKY_ARGS = {
    "I_pow_law": "I_power_law",
    "si_mean": "spec_idx_mean",
    "si_std": "spec_idx_std",
}
# The arguments that `add_astro_sources` passed on before it passed those on as well
OLD_ARGS = {
    "n_src",
    "min_I",
    "max_I",
    "freqs",
    "fov",
    "beam_width",
    "random_seed",
    "n_beam",
}
# Where the observation keeps the intensities of each type of source
SRC_TYPES = {"point": "ast_p_I", "gauss": "ast_g_I", "exp": "ast_e_I"}

# `min_I` defaults to 3sigma, which is above `max_I` in an observation this small
RANDOM = {"n_src": 20, "min_I": 1e-3, "max_I": 1.0, "random_seed": 123456}
NON_DEFAULT = {"I_pow_law": 2.5, "si_mean": -1.0, "si_std": 0.0}


def signature_defaults() -> dict:
    params = inspect.signature(generate_random_sky).parameters
    return {key: params[arg].default for key, arg in SKY_ARGS.items()}


def load_small_obs(tmp_path, ast_sources: dict):
    """A small observation and its config, completed from the base config as
    `sim-vis` does."""
    sim_config = {
        "telescope": {"name": "MeerKAT", "n_ant": 4},
        "observation": {
            "target_name": "target",
            "ra": 27.0,
            "dec": -30.0,
            "start_time_lha": 0.0,
            "int_time": 2.0,
            "n_time": 3,
            "n_int": 2,
            "start_freq": 1.227e9,
            "chan_width": 209e3,
            "n_freq": 3,
            "SEFD": 420,
            "random_seed": 12345,
        },
        "ast_sources": ast_sources,
    }
    config_path = tmp_path / "sim.yaml"
    config_path.write_text(yaml.safe_dump(sim_config))

    sim_config = load_config(str(config_path), config_type="sim")
    tel_def = get_telescope_definitions(sim_config["telescope"]["name"])
    sim_config["telescope"] = deep_update(sim_config["telescope"], tel_def)

    return load_obs(sim_config), sim_config


class SkySpy:
    """Stands in for `generate_random_sky` in `tabsim.config` to record how it is
    called."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return generate_random_sky(**kwargs)


def add_random_sources(tmp_path, src_type: str, random: dict):
    """Add random sources of one type to a small observation. Returns the arguments
    `generate_random_sky` was called with and the intensities (n_src, n_freq) that
    the observation ended up with."""
    obs, sim_config = load_small_obs(tmp_path, {src_type: {"random": random}})

    spy = SkySpy()
    with patch.object(config, "generate_random_sky", spy):
        add_astro_sources(obs, sim_config)

    assert len(spy.calls) == 1
    (I,) = getattr(obs, SRC_TYPES[src_type])
    I = np.asarray(I)
    # Sources are constant in time
    np.testing.assert_array_equal(I, I[:, :1, :] * np.ones_like(I))

    return spy.calls[0], I[:, 0, :]


def spectral_indices(I: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return -np.log(I[:, -1] / I[:, 0]) / np.log(freqs[-1] / freqs[0])


def assert_bit_identical(a, b):
    """The same type, shape and bytes. Equal values are not enough: they allow
    another type and do not tell 0.0 from -0.0."""
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape
    assert a.tobytes() == b.tobytes()


#####################
# generate_random_sky
#####################


def test_power_law_index_sets_the_flux_distribution():
    """Without a maximum flux nothing is redrawn, so the same seed gives the same
    quantiles of two power laws and I / I_min = (1 - x) ** (1 / (1 - alpha))."""
    I_min, alpha, alpha_steep = 1e-3, 1.6, 2.5
    kwargs = dict(n_src=200, freqs=np.array([1e9]), min_I=I_min, max_I=np.inf)

    I, _, _ = generate_random_sky(I_power_law=alpha, random_seed=7, **kwargs)
    I_steep, _, _ = generate_random_sky(
        I_power_law=alpha_steep, random_seed=7, **kwargs
    )
    I, I_steep = np.asarray(I)[:, 0], np.asarray(I_steep)[:, 0]

    assert np.all(I_steep < I)
    np.testing.assert_allclose(
        np.log(I_steep / I_min),
        (1 - alpha) / (1 - alpha_steep) * np.log(I / I_min),
        rtol=1e-9,
    )


def test_spectral_index_mean_and_std_set_the_spectra():
    freqs = np.array([1.0e9, 1.5e9, 2.0e9])
    kwargs = dict(n_src=1000, freqs=freqs, random_seed=11)

    I, _, _ = generate_random_sky(spec_idx_mean=-0.4, spec_idx_std=0.0, **kwargs)
    np.testing.assert_allclose(spectral_indices(np.asarray(I), freqs), -0.4, rtol=1e-9)

    I, _, _ = generate_random_sky(spec_idx_mean=1.5, spec_idx_std=0.5, **kwargs)
    spec_idx = spectral_indices(np.asarray(I), freqs)
    # 5 standard errors of the mean and of the standard deviation of 1000 draws
    assert abs(spec_idx.mean() - 1.5) < 5 * 0.5 / np.sqrt(1000)
    assert abs(spec_idx.std() - 0.5) < 5 * 0.5 / np.sqrt(2 * 1000)


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.0, -1.6, np.nan])
def test_power_law_index_not_above_one_raises(alpha):
    with pytest.raises(ValueError) as err:
        random_power_law(10, alpha=alpha, random_seed=0)

    assert f"alpha = {alpha}" in str(err.value)
    assert "greater than 1" in str(err.value)

    with pytest.raises(ValueError):
        generate_random_sky(n_src=10, freqs=np.array([1e9]), I_power_law=alpha)


#####################
# add_astro_sources
#####################


@pytest.mark.parametrize("src_type", SRC_TYPES)
def test_base_config_defaults_are_those_of_generate_random_sky(src_type):
    """These used to be ignored in favour of the defaults of `generate_random_sky`.
    With equal defaults, the seeded sky of a config that has the default values of
    these three settings does not change."""
    config_dir = str(files("tabsim.data").joinpath("config"))
    base_config = yaml_load(os.path.join(config_dir, "sim_config_base.yaml"))
    rand_ = base_config["ast_sources"][src_type]["random"]

    assert {key: rand_[key] for key in SKY_ARGS} == signature_defaults()


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("src_type", SRC_TYPES)
def test_default_values_leave_seeded_sky_unchanged(tmp_path, src_type, explicit):
    """The default values, left out of the config or written out in it as in the
    example configs, give bit for bit the sky of the call without them."""
    random = dict(RANDOM, **signature_defaults()) if explicit else RANDOM

    kwargs, I = add_random_sources(tmp_path, src_type, random)

    assert set(kwargs) == OLD_ARGS | set(SKY_ARGS.values())
    old_kwargs = {arg: value for arg, value in kwargs.items() if arg in OLD_ARGS}
    I_old, d_ra_old, d_dec_old = generate_random_sky(**old_kwargs)
    I_new, d_ra, d_dec = generate_random_sky(**kwargs)

    assert_bit_identical(I, I_old)
    assert_bit_identical(I_new, I_old)
    assert_bit_identical(d_ra, d_ra_old)
    assert_bit_identical(d_dec, d_dec_old)


@pytest.mark.parametrize("src_type", SRC_TYPES)
def test_config_values_set_the_fluxes_and_spectral_indices(tmp_path, src_type):
    _, I_default = add_random_sources(tmp_path, src_type, RANDOM)
    kwargs, I = add_random_sources(tmp_path, src_type, dict(RANDOM, **NON_DEFAULT))

    assert {key: kwargs[arg] for key, arg in SKY_ARGS.items()} == NON_DEFAULT

    freqs = np.asarray(kwargs["freqs"])
    # The fluxes at the first frequency are those drawn from the power law
    assert not np.any(I[:, 0] == I_default[:, 0])
    assert np.median(I[:, 0]) < np.median(I_default[:, 0])
    assert spectral_indices(I_default, freqs).std() > 0.1
    np.testing.assert_allclose(
        spectral_indices(I, freqs), NON_DEFAULT["si_mean"], rtol=1e-6
    )


@pytest.mark.parametrize("src_type", SRC_TYPES)
@pytest.mark.parametrize("I_pow_law", [1.0, 0.6, -1.6])
def test_config_power_law_index_not_above_one_raises(tmp_path, src_type, I_pow_law):
    obs, sim_config = load_small_obs(
        tmp_path, {src_type: {"random": dict(RANDOM, I_pow_law=I_pow_law)}}
    )

    with pytest.raises(ValueError) as err:
        add_astro_sources(obs, sim_config)

    assert f"ast_sources.{src_type}.random" in str(err.value)
    assert f"I_pow_law = {I_pow_law}" in str(err.value)
    assert "greater than 1" in str(err.value)
    assert getattr(obs, SRC_TYPES[src_type]) == []


@pytest.mark.parametrize("src_type", SRC_TYPES)
def test_config_negative_spectral_index_std_raises(tmp_path, src_type):
    obs, sim_config = load_small_obs(
        tmp_path, {src_type: {"random": dict(RANDOM, si_std=-0.2)}}
    )

    with pytest.raises(ValueError) as err:
        add_astro_sources(obs, sim_config)

    assert f"ast_sources.{src_type}.random" in str(err.value)
    assert "si_std = -0.2" in str(err.value)
    assert "must not be negative" in str(err.value)
    assert getattr(obs, SRC_TYPES[src_type]) == []


# What a config can hold where a number belongs: a number in quotes, a setting left
# empty, a bool, a list, and numbers that are not finite or too large to be a float
NOT_NUMBERS = {
    "quoted": "2.0",
    "empty": None,
    "bool": True,
    "list": [2.0],
    "nan": float("nan"),
    "inf": float("inf"),
    "huge_int": 10**400,
}


@pytest.mark.parametrize("value", list(NOT_NUMBERS.values()), ids=list(NOT_NUMBERS))
@pytest.mark.parametrize("name", SKY_ARGS)
def test_config_value_that_is_not_a_finite_number_raises(tmp_path, name, value):
    """These used to fail, if at all, with an error that did not name the setting."""
    obs, sim_config = load_small_obs(
        tmp_path, {"point": {"random": dict(RANDOM, **{name: value})}}
    )
    # The value has to survive the config file and the defaults of the base config
    loaded = sim_config["ast_sources"]["point"]["random"][name]
    assert type(loaded) is type(value) and repr(loaded) == repr(value)

    with pytest.raises(ValueError) as err:
        add_astro_sources(obs, sim_config)

    assert "ast_sources.point.random" in str(err.value)
    assert f"{name} = {value!r} must be a finite number" in str(err.value)
    assert obs.ast_p_I == []


def test_config_integer_values_are_numbers(tmp_path):
    integers = {"I_pow_law": 2, "si_mean": 1, "si_std": 0}

    kwargs, I = add_random_sources(tmp_path, "point", dict(RANDOM, **integers))

    assert {key: kwargs[arg] for key, arg in SKY_ARGS.items()} == integers
    np.testing.assert_allclose(
        spectral_indices(I, np.asarray(kwargs["freqs"])), 1.0, rtol=1e-6
    )
