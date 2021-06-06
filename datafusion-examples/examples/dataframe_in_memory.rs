// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::sync::Arc;

use datafusion::arrow::array::{Int32Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty;

use datafusion::datasource::MemTable;
use datafusion::error::Result;
use datafusion::prelude::*;

/// This example demonstrates how to use the DataFrame API against in-memory data.
#[tokio::main]
async fn main() -> Result<()> {
    // define a schema.
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8, false),
        Field::new("b", DataType::Int32, false),
    ]));

    // define data.
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            Arc::new(Int32Array::from(vec![1, 10, 10, 100])),
        ],
    )?;

    // declare a new context. In spark API, this corresponds to a new spark SQLsession
    let mut ctx = ExecutionContext::new();

    // declare a table in memory. In spark API, this corresponds to createDataFrame(...).
    let provider = MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table("t", Arc::new(provider))?;
    let df = ctx.table("t")?;

    // Sum column b
    let df =
        df.each_aggregate(vec![], vec![sum(col("b")), sum(col("b")).alias("sum_b")])?;
    let df = df.select_columns(&["SUM(b)", "sum_b"])?;

    // execute
    let results = df.collect().await?;

    // print the results
    pretty::print_batches(&results)?;

    Ok(())
}
