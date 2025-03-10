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
use integer_encoding::{VarInt, VarIntReader, VarIntWriter};
use std::cmp::Ordering;
use std::io::{Read, Write};

use super::error::DecodingError;

pub trait ReadExt: Read {
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
                        values.push(self.read_varint().map_err(DecodingError::Value)?);
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

pub fn write_diff<W: Write, A: VarInt>(
    writer: &mut W,
    diff: &super::Diff<A>,
) -> Result<(), std::io::Error> {
    writer.write_all(&(diff.ops.len() as u32).to_be_bytes())?;
    for op in &diff.ops {
        match op {
            super::Op::Take(len) => {
                writer.write_all(&(*len as i32).to_be_bytes())?;
            }
            super::Op::Drop(len) => {
                writer.write_all(&(-(*len as i32)).to_be_bytes())?;
            }
            super::Op::Insert(values) => {
                writer.write_all(&0_i32.to_be_bytes())?;
                writer.write_all(&(values.len() as u32).to_be_bytes())?;

                for value in values {
                    writer.write_varint(*value)?;
                }
            }
        }
    }

    Ok(())
}
