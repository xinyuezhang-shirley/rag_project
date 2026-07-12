from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DataSourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["生产数据库"])
    db_type: str = Field(..., pattern="^(postgresql|mysql)$")
    host: str = Field(..., examples=["localhost"])
    port: int = Field(..., ge=1, le=65535, examples=[5432])
    database_name: str = Field(..., examples=["mydb"])
    username: str = Field(..., examples=["postgres"])
    password: str = Field(..., min_length=1)


class DataSourceUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100, examples=["生产数据库"])
    host: Optional[str] = Field(None, examples=["localhost"])
    port: Optional[int] = Field(None, ge=1, le=65535, examples=[5432])
    database_name: Optional[str] = Field(None, examples=["mydb"])
    username: Optional[str] = Field(None, examples=["postgres"])
    password: Optional[str] = Field(None, min_length=1)


class DataSourceResponse(BaseModel):
    id: int
    name: str
    db_type: str
    host: str
    port: int
    database_name: str
    username: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DataSourceListResponse(BaseModel):
    items: list[DataSourceResponse]
    next_cursor: Optional[str] = None


class ConnectionTestResponse(BaseModel):
    success: bool
    message: str


class TableSchema(BaseModel):
    table_name: str
    columns: list["ColumnSchema"]


class ColumnSchema(BaseModel):
    column_name: str
    column_type: str
    is_nullable: bool
    comment: Optional[str] = None


class SchemaIntrospectionResponse(BaseModel):
    datasource_id: int
    datasource_name: str
    tables: list[TableSchema]
    table_count: int