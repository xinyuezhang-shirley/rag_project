"""作业1/4 测试：列名授权校验 + 敏感列过滤（app/sql_engine/validator.py）。"""

from app.sql_engine.validator import validate_sql

ALLOWED_TABLES = ["demo_orders"]
ALLOWED_COLUMNS = {
    "demo_orders": ["id", "customer_name", "amount", "category", "created_at"],
}


def test_valid_query_with_known_columns_passes():
    result = validate_sql(
        "SELECT category, SUM(amount) AS total FROM demo_orders GROUP BY category ORDER BY total DESC",
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
    )
    assert result.is_safe


def test_nonexistent_column_is_blocked():
    """作业1 要求的测试用例：故意引用不存在的列名，应被拦截。"""
    result = validate_sql(
        "SELECT nonexistent_col FROM demo_orders",
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
    )
    assert not result.is_safe
    assert result.level == "L2"
    assert "nonexistent_col" in result.blocked_detail


def test_qualified_column_via_table_alias_is_resolved():
    result = validate_sql(
        "SELECT o.category FROM demo_orders o",
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
    )
    assert result.is_safe


def test_select_star_is_not_checked_against_column_list():
    result = validate_sql(
        "SELECT * FROM demo_orders",
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
    )
    assert result.is_safe


def test_order_by_referencing_select_alias_is_not_flagged():
    result = validate_sql(
        "SELECT amount AS total_sales FROM demo_orders ORDER BY total_sales",
        allowed_tables=ALLOWED_TABLES,
        allowed_columns=ALLOWED_COLUMNS,
    )
    assert result.is_safe


def test_no_allowed_columns_skips_column_check():
    result = validate_sql(
        "SELECT nonexistent_col FROM demo_orders",
        allowed_tables=ALLOWED_TABLES,
    )
    assert result.is_safe


def test_sensitive_column_is_blocked():
    """作业4：查询敏感列应被拦截并提示原因。"""
    result = validate_sql(
        "SELECT password FROM users",
        allowed_tables=["users"],
        sensitive_columns=["password", "ssn"],
    )
    assert not result.is_safe
    assert result.level == "L5"
    assert "password" in result.reason


def test_sensitive_keyword_used_only_as_output_alias_is_not_blocked():
    result = validate_sql(
        "SELECT amount AS password FROM demo_orders",
        allowed_tables=ALLOWED_TABLES,
        sensitive_columns=["password"],
    )
    assert result.is_safe


def test_no_sensitive_columns_configured_skips_check():
    result = validate_sql(
        "SELECT password FROM users",
        allowed_tables=["users"],
    )
    assert result.is_safe
