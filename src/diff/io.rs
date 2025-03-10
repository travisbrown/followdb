//! Convert diffs to a binary representation.
//!
//! The representation begins with a `u32` value indicating the number of operations in the diff.
//! The `Take` and `Drop` operations are represented by a single positive or negative non-zero
//! `i32` number respectively. The `Insert` operation begins with a zero, followed by a `u32`
//! representing the number of inserted values, followed by the values themselves, using a varint
//! encoding.
//!
//! The big-endian encoding is used for all `u32` and `i32` numbers.

use byteorder::{BigEndian, ReadBytesExt};
use chrono::{TimeZone, Utc};
use integer_encoding::{VarInt, VarIntReader, VarIntWriter};
use std::cmp::Ordering;
use std::io::{Read, Write};

use super::error::DecodingError;

pub trait ReadExt: Read + VarIntReader {
    fn read_update(&mut self) -> Result<super::Update<u64>, DecodingError> {
        let timestamp = self
            .read_u32::<BigEndian>()
            .map_err(DecodingError::Timestamp)
            .and_then(|timestamp_s| {
                Utc.timestamp_opt(timestamp_s.into(), 0)
                    .single()
                    .ok_or(DecodingError::InvalidTimestamp(timestamp_s))
            })?;
        let id = self.read_u64::<BigEndian>().map_err(DecodingError::Id)?;
        let diff = self.read_diff()?;

        Ok(super::Update::new(timestamp, id, diff))
    }

    fn read_diff<A: VarInt>(&mut self) -> Result<super::Diff<A>, DecodingError> {
        let count = self.read_u32::<BigEndian>().map_err(DecodingError::Count)?;

        let mut ops = Vec::with_capacity(count as usize);
        let mut drop_count = 0;
        let mut len_change = 0;

        for _ in 0..count {
            let next = self.read_i32::<BigEndian>().map_err(DecodingError::Value)?;

            match next.cmp(&0) {
                Ordering::Greater => {
                    ops.push(super::Op::Take(next as usize));
                }
                Ordering::Less => {
                    ops.push(super::Op::Drop((-next) as usize));
                    drop_count += 1;
                    len_change += next as isize;
                }
                Ordering::Equal => {
                    let insert_count = self
                        .read_u32::<BigEndian>()
                        .map_err(DecodingError::InsertCount)?;

                    let mut values = Vec::with_capacity(insert_count as usize);

                    for _ in 0..insert_count {
                        values.push(self.read_varint::<A>().map_err(DecodingError::Value)?);
                    }

                    ops.push(super::Op::Insert(values));
                    len_change += insert_count as isize;
                }
            }
        }

        Ok(super::Diff {
            ops,
            len_change,
            drop_count,
        })
    }
}

impl<R: Read + VarIntReader> ReadExt for R {}

pub trait WriteExt: Write + VarIntWriter {
    fn write_update(&mut self, update: &super::Update<u64>) -> Result<(), std::io::Error> {
        self.write_all(&(update.timestamp.timestamp() as u32).to_be_bytes())?;
        self.write_all(&update.id.to_be_bytes())?;
        self.write_diff(&update.diff)?;

        Ok(())
    }

    fn write_diff<A: VarInt>(&mut self, diff: &super::Diff<A>) -> Result<(), std::io::Error> {
        self.write_all(&(diff.ops.len() as u32).to_be_bytes())?;
        for op in &diff.ops {
            match op {
                super::Op::Take(len) => {
                    self.write_all(&(*len as i32).to_be_bytes())?;
                }
                super::Op::Drop(len) => {
                    self.write_all(&(-(*len as i32)).to_be_bytes())?;
                }
                super::Op::Insert(values) => {
                    self.write_all(&0_i32.to_be_bytes())?;
                    self.write_all(&(values.len() as u32).to_be_bytes())?;

                    for value in values {
                        self.write_varint(*value)?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl<W: Write + VarIntWriter> WriteExt for W {}
