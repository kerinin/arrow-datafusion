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

//! Defines the execution plan for the hash aggregate operation

use std::any::Any;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::vec;

use ahash::RandomState;
use futures::stream::{Stream, StreamExt};

use crate::error::{DataFusionError, Result};
use crate::physical_plan::{
    Accumulator, AggregateExpr, DisplayFormatType, Distribution, ExecutionPlan,
    Partitioning, PhysicalExpr, SQLMetric,
};
use crate::scalar::ScalarValue;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::{array::Array, error::Result as ArrowResult};
use arrow::{
    array::{BooleanArray, Date32Array, DictionaryArray},
    datatypes::{
        ArrowDictionaryKeyType, ArrowNativeType, Int16Type, Int32Type, Int64Type,
        Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    },
};
use arrow::{
    datatypes::{DataType, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use hashbrown::HashMap;
use ordered_float::OrderedFloat;
use pin_project_lite::pin_project;

use arrow::array::{
    LargeStringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray,
};
use async_trait::async_trait;

use super::{group_scalar::GroupByScalar, RecordBatchStream, SendableRecordBatchStream};

/// Hash aggregate execution plan
#[derive(Debug)]
pub struct HashEachAggregateExec {
    /// Grouping expressions
    group_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
    /// Aggregate expressions
    aggr_expr: Vec<Arc<dyn AggregateExpr>>,
    /// Input plan, could be a partial aggregate or the input to the aggregate
    input: Arc<dyn ExecutionPlan>,
    /// Schema after the aggregate is applied
    schema: SchemaRef,
    /// Input schema before any aggregation is applied. For partial aggregate this will be the
    /// same as input.schema() but for the final aggregate it will be the same as the input
    /// to the partial aggregate
    input_schema: SchemaRef,
    /// Metric to track number of output rows
    output_rows: Arc<SQLMetric>,
}

fn create_schema(
    input_schema: &Schema,
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Schema> {
    let mut fields = Vec::with_capacity(input_schema.fields().len() + aggr_expr.len());
    for field in input_schema.fields() {
        fields.push(field.clone())
    }
    for expr in aggr_expr {
        fields.push(expr.field()?)
    }

    Ok(Schema::new(fields))
}

impl HashEachAggregateExec {
    /// Create a new hash aggregate execution plan
    pub fn try_new(
        group_expr: Vec<(Arc<dyn PhysicalExpr>, String)>,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        input: Arc<dyn ExecutionPlan>,
        input_schema: SchemaRef,
    ) -> Result<Self> {
        let schema = create_schema(&input.schema(), &aggr_expr)?;

        let schema = Arc::new(schema);

        let output_rows = SQLMetric::counter();

        Ok(HashEachAggregateExec {
            group_expr,
            aggr_expr,
            input,
            schema,
            input_schema,
            output_rows,
        })
    }

    /// Grouping expressions
    pub fn group_expr(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.group_expr
    }

    /// Aggregate expressions
    pub fn aggr_expr(&self) -> &[Arc<dyn AggregateExpr>] {
        &self.aggr_expr
    }

    /// Input plan
    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    /// Get the input schema before any aggregates are applied
    pub fn input_schema(&self) -> SchemaRef {
        self.input_schema.clone()
    }
}

#[async_trait]
impl ExecutionPlan for HashEachAggregateExec {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn children(&self) -> Vec<Arc<dyn ExecutionPlan>> {
        vec![self.input.clone()]
    }

    fn required_child_distribution(&self) -> Distribution {
        if self.group_expr.is_empty() {
            Distribution::SinglePartition
        } else {
            Distribution::HashPartitioned(
                self.group_expr.iter().map(|x| x.0.clone()).collect(),
            )
        }
    }

    /// Get the output partitioning of this plan
    fn output_partitioning(&self) -> Partitioning {
        self.input.output_partitioning()
    }

    async fn execute(&self, partition: usize) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition).await?;
        let group_expr = self.group_expr.iter().map(|x| x.0.clone()).collect();

        let aggr_expr_copy = self.aggr_expr.clone();
        let accumulators = create_accumulators(&aggr_expr_copy)
            .map_err(DataFusionError::into_arrow_external_error)?;

        let expressions = aggregate_expressions(&aggr_expr_copy)
            .map_err(DataFusionError::into_arrow_external_error)?;

        if self.group_expr.is_empty() {
            Ok(Box::pin(HashEachAggregateStream {
                schema: self.schema.clone(),
                accumulators: accumulators,
                expressions: expressions,
                input: input,
            }))
        } else {
            Ok(Box::pin(GroupedHashEachAggregateStream {
                schema: self.schema.clone(),
                group_expr: group_expr,
                aggr_expr: self.aggr_expr.clone(),
                accumulators: accumulators,
                expressions: expressions,
                input: input,
                output_rows: self.output_rows.clone(),
            }))
        }
    }

    fn with_new_children(
        &self,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        match children.len() {
            1 => Ok(Arc::new(HashEachAggregateExec::try_new(
                self.group_expr.clone(),
                self.aggr_expr.clone(),
                children[0].clone(),
                self.input_schema.clone(),
            )?)),
            _ => Err(DataFusionError::Internal(
                "HashEachAggregateExec wrong number of children".to_string(),
            )),
        }
    }

    fn metrics(&self) -> HashMap<String, SQLMetric> {
        let mut metrics = HashMap::new();
        metrics.insert("outputRows".to_owned(), (*self.output_rows).clone());
        metrics
    }

    fn fmt_as(
        &self,
        t: DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default => {
                write!(f, "HashEachAggregateExec: ")?;
                let g: Vec<String> = self
                    .group_expr
                    .iter()
                    .map(|(e, alias)| {
                        let e = e.to_string();
                        if &e != alias {
                            format!("{} as {}", e, alias)
                        } else {
                            e
                        }
                    })
                    .collect();
                write!(f, ", gby=[{}]", g.join(", "))?;

                let a: Vec<String> = self
                    .aggr_expr
                    .iter()
                    .map(|agg| agg.name().to_string())
                    .collect();
                write!(f, ", aggr=[{}]", a.join(", "))?;
            }
        }
        Ok(())
    }
}

pin_project! {
    struct GroupedHashEachAggregateStream {
        schema: SchemaRef,
        #[pin]
        group_expr: Vec<Arc<dyn PhysicalExpr>>,
        aggr_expr: Vec<Arc<dyn AggregateExpr>>,
        accumulators: Vec<AccumulatorItem>,
        expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
        input: SendableRecordBatchStream,
        output_rows: Arc<SQLMetric>,
    }
}

fn group_aggregate_batch(
    group_expr: &[Arc<dyn PhysicalExpr>],
    aggr_expr: &[Arc<dyn AggregateExpr>],
    batch: RecordBatch,
    schema: SchemaRef,
    mut accumulators: Accumulators,
    aggregate_expressions: &[Vec<Arc<dyn PhysicalExpr>>],
) -> ArrowResult<RecordBatch> {
    // evaluate the grouping expressions
    let group_values: Vec<ArrayRef> = evaluate(group_expr, &batch)
        .map_err(DataFusionError::into_arrow_external_error)?;

    // create vector large enough to hold the grouping key
    // this is an optimization to avoid allocating `key` on every row.
    // it will be overwritten on every iteration of the loop below
    // NOTE: Vector containing a GroupByScalar associated with each group expr for a single row (the current row)
    let mut group_by_values = Vec::with_capacity(group_values.len());
    for _ in 0..group_values.len() {
        group_by_values.push(GroupByScalar::UInt32(0));
    }
    let mut group_by_values = group_by_values.into_boxed_slice();

    // Vector of u8.
    // Group-val len is a lower bound, this gets extended with the content of each group-by value.
    let mut key: Vec<u8> = Vec::with_capacity(group_values.len());

    let accumulator_columns: Vec<ArrayRef> = (0..aggregate_expressions.len())
        .zip(aggregate_expressions)
        .map(|(accum_idx, exprs)| {
            // Compute the result of each expression over the batch
            let expr_values = exprs
                .iter()
                .map(|e| e.evaluate(&batch).map(|r| r.into_array(batch.num_rows())))
                .collect::<Result<Vec<ArrayRef>>>()?;

            // Iterate over each element of the batch
            let accum_results = (0..expr_values[0].len())
                .map(|row| {
                    // Copy the contents of the row's group_values into the key byte vector
                    create_key(&group_values, row, &mut key)
                        .map_err(DataFusionError::into_arrow_external_error)?;

                    // TODO: Refactor this to do a single pass building an array of accumulators and assigning
                    // offsets in that array to each row. Then we can looukp accumulators by offset for each
                    // aggregation expression, rather than doing this (more expensive) hash lookup.
                    let (_, group_accums) = accumulators
                        .raw_entry_mut()
                        .from_key(&key)
                        .or_insert_with(|| {
                            // We can safely unwrap here as we checked we can create an accumulator before
                            let accumulator_set = create_accumulators(aggr_expr).unwrap();

                            let _ = create_group_by_values(
                                &group_values,
                                row,
                                &mut group_by_values,
                            );

                            (key.clone(), accumulator_set)
                        });
                    let accum = &mut group_accums[accum_idx];

                    // Build an ArrayRef containing the value of each expression
                    let v = expr_values
                        .iter()
                        .map(|array| ScalarValue::try_from_array(array, row))
                        .collect::<Result<Vec<_>>>()?;

                    // Update the accumulator with the value of each expression at the current offset
                    accum.update(&v)?;
                    // ...and return the resulting accumulator value
                    accum.evaluate()
                })
                .collect::<Result<Vec<ScalarValue>>>()?;

            // Convert the value of the accumulator associated with each element of the batch into an array
            ScalarValue::iter_to_array(accum_results)
        })
        .collect::<Result<Vec<ArrayRef>>>()
        .map_err(DataFusionError::into_arrow_external_error)?;

    RecordBatch::try_new(
        schema,
        batch
            .columns()
            .iter()
            .map(|c| c.clone())
            .chain(accumulator_columns.into_iter())
            .collect::<Vec<ArrayRef>>(),
    )
}

/// Appends a sequence of [u8] bytes for the value in `col[row]` to
/// `vec` to be used as a key into the hash map for a dictionary type
///
/// Note that ideally, for dictionary encoded columns, we would be
/// able to simply use the dictionary idicies themselves (no need to
/// look up values) or possibly simply build the hash table entirely
/// on the dictionary indexes.
///
/// This aproach would likely work (very) well for the common case,
/// but it also has to to handle the case where the dictionary itself
/// is not the same across all record batches (and thus indexes in one
/// record batch may not correspond to the same index in another)
fn dictionary_create_key_for_col<K: ArrowDictionaryKeyType>(
    col: &ArrayRef,
    row: usize,
    vec: &mut Vec<u8>,
) -> Result<()> {
    let dict_col = col.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();

    // look up the index in the values dictionary
    let keys_col = dict_col.keys_array();
    let values_index = keys_col.value(row).to_usize().ok_or_else(|| {
        DataFusionError::Internal(format!(
            "Can not convert index to usize in dictionary of type creating group by value {:?}",
            keys_col.data_type()
        ))
    })?;

    create_key_for_col(&dict_col.values(), values_index, vec)
}

/// Appends a sequence of [u8] bytes for the value in `col[row]` to
/// `vec` to be used as a key into the hash map
fn create_key_for_col(col: &ArrayRef, row: usize, vec: &mut Vec<u8>) -> Result<()> {
    match col.data_type() {
        DataType::Boolean => {
            let array = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            vec.extend_from_slice(&[array.value(row) as u8]);
        }
        DataType::Float32 => {
            let array = col.as_any().downcast_ref::<Float32Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Float64 => {
            let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::UInt8 => {
            let array = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::UInt16 => {
            let array = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::UInt32 => {
            let array = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::UInt64 => {
            let array = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Int8 => {
            let array = col.as_any().downcast_ref::<Int8Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Int16 => {
            let array = col.as_any().downcast_ref::<Int16Array>().unwrap();
            vec.extend(array.value(row).to_le_bytes().iter());
        }
        DataType::Int32 => {
            let array = col.as_any().downcast_ref::<Int32Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Int64 => {
            let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Utf8 => {
            let array = col.as_any().downcast_ref::<StringArray>().unwrap();
            let value = array.value(row);
            // store the size
            vec.extend_from_slice(&value.len().to_le_bytes());
            // store the string value
            vec.extend_from_slice(value.as_bytes());
        }
        DataType::LargeUtf8 => {
            let array = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            let value = array.value(row);
            // store the size
            vec.extend_from_slice(&value.len().to_le_bytes());
            // store the string value
            vec.extend_from_slice(value.as_bytes());
        }
        DataType::Date32 => {
            let array = col.as_any().downcast_ref::<Date32Array>().unwrap();
            vec.extend_from_slice(&array.value(row).to_le_bytes());
        }
        DataType::Dictionary(index_type, _) => match **index_type {
            DataType::Int8 => {
                dictionary_create_key_for_col::<Int8Type>(col, row, vec)?;
            }
            DataType::Int16 => {
                dictionary_create_key_for_col::<Int16Type>(col, row, vec)?;
            }
            DataType::Int32 => {
                dictionary_create_key_for_col::<Int32Type>(col, row, vec)?;
            }
            DataType::Int64 => {
                dictionary_create_key_for_col::<Int64Type>(col, row, vec)?;
            }
            DataType::UInt8 => {
                dictionary_create_key_for_col::<UInt8Type>(col, row, vec)?;
            }
            DataType::UInt16 => {
                dictionary_create_key_for_col::<UInt16Type>(col, row, vec)?;
            }
            DataType::UInt32 => {
                dictionary_create_key_for_col::<UInt32Type>(col, row, vec)?;
            }
            DataType::UInt64 => {
                dictionary_create_key_for_col::<UInt64Type>(col, row, vec)?;
            }
            _ => {
                return Err(DataFusionError::Internal(format!(
                "Unsupported GROUP BY type (dictionary index type not supported creating key) {}",
                col.data_type(),
            )))
            }
        },
        _ => {
            // This is internal because we should have caught this before.
            return Err(DataFusionError::Internal(format!(
                "Unsupported GROUP BY type creating key {}",
                col.data_type(),
            )));
        }
    }
    Ok(())
}

/// Create a key `Vec<u8>` that is used as key for the hashmap
pub(crate) fn create_key(
    group_by_keys: &[ArrayRef],
    row: usize,
    vec: &mut Vec<u8>,
) -> Result<()> {
    vec.clear();
    for col in group_by_keys {
        create_key_for_col(col, row, vec)?
    }
    Ok(())
}

type AccumulatorItem = Box<dyn Accumulator>;
// NOTE: (value of each group-by expression for a row, the accumulator set for the row, indices in the batch with the given group-by value)
type Accumulators = HashMap<Vec<u8>, Vec<AccumulatorItem>, RandomState>;

impl Stream for GroupedHashEachAggregateStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // mapping key -> (set of accumulators, indices of the key in the batch)
        // * the indexes are updated at each row
        // * the accumulators are updated at the end of each batch
        // * the indexes are `clear`ed at the end of each batch
        //let mut accumulators: Accumulators = FnvHashMap::default();

        // iterate over all input batches and update the accumulators
        let accumulators = Accumulators::default();

        // TODO: Cloning on each poll sort of sucks, but we need a mutable ref to accumulators.
        let expressions = &self.expressions.clone();

        self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(group_aggregate_batch(
                &self.group_expr,
                &self.aggr_expr,
                batch,
                self.schema.clone(),
                accumulators,
                &expressions,
            )),
            other => other,
        })
    }
}

impl RecordBatchStream for GroupedHashEachAggregateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Evaluates expressions against a record batch.
fn evaluate(
    expr: &[Arc<dyn PhysicalExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    expr.iter()
        .map(|expr| expr.evaluate(&batch))
        .map(|r| r.map(|v| v.into_array(batch.num_rows())))
        .collect::<Result<Vec<_>>>()
}

/// returns physical expressions to evaluate against a batch
/// The return value is to be understood as:
/// * index 0 is the aggregation
/// * index 1 is the expression i of the aggregation
fn aggregate_expressions(
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Vec<Vec<Arc<dyn PhysicalExpr>>>> {
    Ok(aggr_expr.iter().map(|agg| agg.expressions()).collect())
}

pin_project! {
    /// stream struct for hash aggregation
    pub struct HashEachAggregateStream {
        schema: SchemaRef,
        #[pin]
        accumulators: Vec<AccumulatorItem>,
        expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
        input: SendableRecordBatchStream,
    }
}

fn aggregate_batch(
    batch: &RecordBatch,
    schema: SchemaRef,
    accumulators: &mut [AccumulatorItem],
    expressions: &[Vec<Arc<dyn PhysicalExpr>>],
) -> ArrowResult<RecordBatch> {
    let accumulator_columns = accumulators
        .iter_mut()
        .zip(expressions)
        .map(|(accum, exprs)| {
            // Compute the result of each expression over the batch
            let expr_values = exprs
                .iter()
                .map(|e| e.evaluate(batch).map(|r| r.into_array(batch.num_rows())))
                .collect::<Result<Vec<ArrayRef>>>()?;

            // Iterate over each element of the batch
            let accum_results = (0..expr_values[0].len())
                .map(|index| {
                    // ...and build an ArrayRef containing the value of each expression
                    let v = expr_values
                        .iter()
                        .map(|array| ScalarValue::try_from_array(array, index))
                        .collect::<Result<Vec<_>>>()?;

                    // Update the accumulator with the value of each expression at the current offset
                    accum.update(&v)?;
                    // ...and return the resulting accumulator value
                    accum.evaluate()
                })
                .collect::<Result<Vec<ScalarValue>>>()?;

            // Convert the value of the accumulator associated with each element of the batch into an array
            ScalarValue::iter_to_array(accum_results)
        })
        .collect::<Result<Vec<ArrayRef>>>()
        .map_err(DataFusionError::into_arrow_external_error)?;

    RecordBatch::try_new(
        schema,
        batch
            .columns()
            .iter()
            .map(|c| c.clone())
            .chain(accumulator_columns.into_iter())
            .collect::<Vec<ArrayRef>>(),
    )
}

impl Stream for HashEachAggregateStream {
    type Item = ArrowResult<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // TODO: Cloning on each poll sort of sucks, but we need a mutable ref to accumulators.
        let expressions = &self.expressions.clone();

        self.input.poll_next_unpin(cx).map(|x| match x {
            Some(Ok(batch)) => Some(aggregate_batch(
                &batch,
                self.schema.clone(),
                &mut self.accumulators,
                expressions,
            )),
            other => other,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // same number of record batches
        self.input.size_hint()
    }
}

impl RecordBatchStream for HashEachAggregateStream {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

fn create_accumulators(
    aggr_expr: &[Arc<dyn AggregateExpr>],
) -> Result<Vec<AccumulatorItem>> {
    aggr_expr
        .iter()
        .map(|expr| expr.create_accumulator())
        .collect::<Result<Vec<_>>>()
}

/// Extract the value in `col[row]` from a dictionary a GroupByScalar
fn dictionary_create_group_by_value<K: ArrowDictionaryKeyType>(
    col: &ArrayRef,
    row: usize,
) -> Result<GroupByScalar> {
    let dict_col = col.as_any().downcast_ref::<DictionaryArray<K>>().unwrap();

    // look up the index in the values dictionary
    let keys_col = dict_col.keys_array();
    let values_index = keys_col.value(row).to_usize().ok_or_else(|| {
        DataFusionError::Internal(format!(
            "Can not convert index to usize in dictionary of type creating group by value {:?}",
            keys_col.data_type()
        ))
    })?;

    create_group_by_value(&dict_col.values(), values_index)
}

/// Extract the value in `col[row]` as a GroupByScalar
fn create_group_by_value(col: &ArrayRef, row: usize) -> Result<GroupByScalar> {
    match col.data_type() {
        DataType::Float32 => {
            let array = col.as_any().downcast_ref::<Float32Array>().unwrap();
            Ok(GroupByScalar::Float32(OrderedFloat::from(array.value(row))))
        }
        DataType::Float64 => {
            let array = col.as_any().downcast_ref::<Float64Array>().unwrap();
            Ok(GroupByScalar::Float64(OrderedFloat::from(array.value(row))))
        }
        DataType::UInt8 => {
            let array = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            Ok(GroupByScalar::UInt8(array.value(row)))
        }
        DataType::UInt16 => {
            let array = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            Ok(GroupByScalar::UInt16(array.value(row)))
        }
        DataType::UInt32 => {
            let array = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            Ok(GroupByScalar::UInt32(array.value(row)))
        }
        DataType::UInt64 => {
            let array = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            Ok(GroupByScalar::UInt64(array.value(row)))
        }
        DataType::Int8 => {
            let array = col.as_any().downcast_ref::<Int8Array>().unwrap();
            Ok(GroupByScalar::Int8(array.value(row)))
        }
        DataType::Int16 => {
            let array = col.as_any().downcast_ref::<Int16Array>().unwrap();
            Ok(GroupByScalar::Int16(array.value(row)))
        }
        DataType::Int32 => {
            let array = col.as_any().downcast_ref::<Int32Array>().unwrap();
            Ok(GroupByScalar::Int32(array.value(row)))
        }
        DataType::Int64 => {
            let array = col.as_any().downcast_ref::<Int64Array>().unwrap();
            Ok(GroupByScalar::Int64(array.value(row)))
        }
        DataType::Utf8 => {
            let array = col.as_any().downcast_ref::<StringArray>().unwrap();
            Ok(GroupByScalar::Utf8(Box::new(array.value(row).into())))
        }
        DataType::LargeUtf8 => {
            let array = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Ok(GroupByScalar::LargeUtf8(Box::new(array.value(row).into())))
        }
        DataType::Boolean => {
            let array = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok(GroupByScalar::Boolean(array.value(row)))
        }
        DataType::Timestamp(TimeUnit::Millisecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampMillisecondArray>()
                .unwrap();
            Ok(GroupByScalar::TimeMillisecond(array.value(row)))
        }
        DataType::Timestamp(TimeUnit::Microsecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            Ok(GroupByScalar::TimeMicrosecond(array.value(row)))
        }
        DataType::Timestamp(TimeUnit::Nanosecond, None) => {
            let array = col
                .as_any()
                .downcast_ref::<TimestampNanosecondArray>()
                .unwrap();
            Ok(GroupByScalar::TimeNanosecond(array.value(row)))
        }
        DataType::Date32 => {
            let array = col.as_any().downcast_ref::<Date32Array>().unwrap();
            Ok(GroupByScalar::Date32(array.value(row)))
        }
        DataType::Dictionary(index_type, _) => match **index_type {
            DataType::Int8 => dictionary_create_group_by_value::<Int8Type>(col, row),
            DataType::Int16 => dictionary_create_group_by_value::<Int16Type>(col, row),
            DataType::Int32 => dictionary_create_group_by_value::<Int32Type>(col, row),
            DataType::Int64 => dictionary_create_group_by_value::<Int64Type>(col, row),
            DataType::UInt8 => dictionary_create_group_by_value::<UInt8Type>(col, row),
            DataType::UInt16 => dictionary_create_group_by_value::<UInt16Type>(col, row),
            DataType::UInt32 => dictionary_create_group_by_value::<UInt32Type>(col, row),
            DataType::UInt64 => dictionary_create_group_by_value::<UInt64Type>(col, row),
            _ => Err(DataFusionError::NotImplemented(format!(
                "Unsupported GROUP BY type (dictionary index type not supported) {}",
                col.data_type(),
            ))),
        },
        _ => Err(DataFusionError::NotImplemented(format!(
            "Unsupported GROUP BY type {}",
            col.data_type(),
        ))),
    }
}

/// Extract the values in `group_by_keys` arrow arrays into the target vector
/// as GroupByScalar values
pub(crate) fn create_group_by_values(
    group_by_keys: &[ArrayRef],
    row: usize,
    vec: &mut Box<[GroupByScalar]>,
) -> Result<()> {
    for (i, col) in group_by_keys.iter().enumerate() {
        vec[i] = create_group_by_value(col, row)?
    }
    Ok(())
}
