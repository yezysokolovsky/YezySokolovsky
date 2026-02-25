import pandas as pd
from typing import Optional, Dict


class MissingDataMixin:

    def count_missing_values(self, df: pd.DataFrame) -> pd.Series:
        """
        Подсчёт количества пропущенных значений по каждому столбцу
        """
        return df.isna().sum()

    def missing_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Формирование отчёта по пропущенным значениям
        """
        total_missing = df.isna().sum()
        percent_missing = (df.isna().sum() / len(df)) * 100

        report = pd.DataFrame({
            "missing_count": total_missing,
            "missing_percent": percent_missing
        })

        report = report[report["missing_count"] > 0]
        return report.sort_values(by="missing_percent", ascending=False)

    def fill_missing(
        self,
        df: pd.DataFrame,
        strategy: str = "mean",
        columns: Optional[list] = None,
        constant_value: Optional[Dict[str, any]] = None
    ) -> pd.DataFrame:
        """
        Заполнение пропущенных значений
        strategy: mean | median | mode | constant
        """

        df = df.copy()

        if columns is None:
            columns = df.columns

        for col in columns:
            if df[col].isna().sum() == 0:
                continue

            if strategy == "mean":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)

            elif strategy == "median":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].median(), inplace=True)

            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)

            elif strategy == "constant":
                if constant_value and col in constant_value:
                    df[col].fillna(constant_value[col], inplace=True)
                else:
                    df[col].fillna(0, inplace=True)

            else:
                raise ValueError("Неизвестная стратегия заполнения")

        return df