import jux.config
from jux.config import EnvConfig, LuxEnvConfig, LuxUnitConfig, UnitConfig


class TestUnitConfig:

    def test_from_to_lux(self):
        lux_config = LuxUnitConfig()
        assert lux_config == UnitConfig.from_lux(lux_config).to_lux()

        jux_config = UnitConfig()
        assert jux_config == UnitConfig.from_lux(jux_config.to_lux())

    def test_default(self):
        assert LuxUnitConfig() == UnitConfig().to_lux()
        assert UnitConfig() == UnitConfig.from_lux(LuxUnitConfig())


class TestEnvConfig:

    def test_from_to_lux(self):
        lux_config = LuxEnvConfig()
        assert lux_config == EnvConfig.from_lux(lux_config).to_lux()

        jux_config = EnvConfig()
        assert jux_config == EnvConfig.from_lux(jux_config.to_lux())

    def test_default(self):
        assert LuxEnvConfig() == EnvConfig().to_lux()
        assert EnvConfig() == EnvConfig.from_lux(LuxEnvConfig())
