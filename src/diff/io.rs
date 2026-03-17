//! Convert diffs and updates to a binary representation.
//!
//! The diff representation begins with a `u32` value indicating the number of operations in the
//! diff. The `Take` and `Drop` operations are represented by a single positive or negative
//! non-zero `i32` number respectively. The `Insert` operation begins with a zero, followed by a
//! `u32` representing the number of inserted values, followed by the values themselves, using a
//! varint encoding.
//!
//! The update representation begins with the timestamp (as a `u32` value, using the big-endian
//! encoding for the sake of sorting) and account ID, and then continues with the diff
//! representation described above.
//!
//! The big-endian encoding is used for all `u32` and `i32` numbers.
//!
//! Note that the update functions are not generic in the ID type, but are only implemented for
//! `u64`. This is because we want to be able to sort the resulting byte strings, which requires a
//! fixed-width big-endian representation. It would be possible to implement these functions for
//! other integer types, but that is not done here (yet).

use byteorder::{BigEndian, ReadBytesExt};
use chrono::{TimeZone, Utc};
use integer_encoding::{VarInt, VarIntReader, VarIntWriter};
use std::cmp::Ordering;
use std::io::{Read, Write};

use super::error::DecodingError;

pub trait ReadExt: Read + VarIntReader {
    /// Read the binary representation of an update.
    ///
    /// Note that this function fixes the ID to `u64`.
    ///
    /// # Errors
    ///
    /// Returns a [`DecodingError`] if the timestamp, identifier, or diff cannot be read or
    /// decoded.
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

    /// Read the binary representation of a diff.
    ///
    /// # Errors
    ///
    /// Returns a [`DecodingError`] if the operation count or operation values cannot be read.
    fn read_diff<A: VarInt>(&mut self) -> Result<super::Diff<A>, DecodingError> {
        let count = self.read_u32::<BigEndian>().map_err(DecodingError::Count)?;

        let mut ops = Vec::with_capacity(count as usize);
        let mut expected_source_len: u64 = 0;
        let mut len_change: i64 = 0;
        let mut drop_count = 0;

        for _ in 0..count {
            let next = self.read_i32::<BigEndian>().map_err(DecodingError::Value)?;

            match next.cmp(&0) {
                Ordering::Greater => {
                    // `next > 0`, so `unsigned_abs() == next as u32`, safe to widen to `u64`.
                    let len = u64::from(next.unsigned_abs());
                    ops.push(super::Op::Take(len));
                    expected_source_len += len;
                }
                Ordering::Less => {
                    // `next < 0`, so `unsigned_abs() == (-next) as u32`, safe to widen to `u64`.
                    let len = u64::from(next.unsigned_abs());
                    ops.push(super::Op::Drop(len));
                    expected_source_len += len;
                    // Widening signed cast is always lossless.
                    len_change += i64::from(next);
                    drop_count += 1;
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
                    // `u32` always fits in `i64`.
                    len_change += i64::from(insert_count);
                }
            }
        }

        Ok(super::Diff {
            ops,
            expected_source_len,
            len_change,
            drop_count,
        })
    }
}

impl<R: Read + VarIntReader> ReadExt for R {}

pub trait WriteExt: Write + VarIntWriter {
    /// Write the binary representation of an update.
    ///
    /// Note that this function fixes the ID to `u64`.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if the timestamp falls outside the `u32` seconds range
    /// (i.e. before 1970 or after 2106), or if any write to the underlying writer fails.
    fn write_update(&mut self, update: &super::Update<u64>) -> Result<(), std::io::Error> {
        let timestamp_u32 = u32::try_from(update.timestamp.timestamp())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.write_all(&timestamp_u32.to_be_bytes())?;
        self.write_all(&update.id.to_be_bytes())?;
        self.write_diff(&update.diff)?;

        Ok(())
    }

    /// Write the binary representation of a diff.
    ///
    /// # Errors
    ///
    /// Returns an [`std::io::Error`] if any op count or length exceeds format bounds, or if any
    /// write to the underlying writer fails.
    fn write_diff<A: VarInt>(&mut self, diff: &super::Diff<A>) -> Result<(), std::io::Error> {
        let op_count = u32::try_from(diff.ops.len())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.write_all(&op_count.to_be_bytes())?;
        for op in &diff.ops {
            match op {
                super::Op::Take(len) => {
                    let len_i32 = i32::try_from(*len)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    self.write_all(&len_i32.to_be_bytes())?;
                }
                super::Op::Drop(len) => {
                    let neg = -i32::try_from(*len)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    self.write_all(&neg.to_be_bytes())?;
                }
                super::Op::Insert(values) => {
                    self.write_all(&0_i32.to_be_bytes())?;
                    let insert_count = u32::try_from(values.len())
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                    self.write_all(&insert_count.to_be_bytes())?;

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
