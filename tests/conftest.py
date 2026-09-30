import pandas as pd
import pytest

from aikeiba.synthetic import generate_runners


@pytest.fixture(scope="session")
def small_runners() -> pd.DataFrame:
    return generate_runners(n_days=150, races_per_day=6, n_horses=600, seed=1)
