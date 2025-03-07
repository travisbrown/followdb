use integer_encoding::{VarIntReader, VarIntWriter};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use zstd::stream::{read::Decoder, write::Encoder};

pub enum IdReader<R> {
    Reading { underlying: R },
    Failed { error: Option<std::io::Error> },
}

impl IdReader<BufReader<Decoder<'static, BufReader<File>>>> {
    pub fn open_zst_file<P: AsRef<Path>>(input: P) -> Self {
        match File::open(input).and_then(zstd::stream::read::Decoder::new) {
            Ok(reader) => Self::Reading {
                underlying: BufReader::new(reader),
            },
            Err(error) => Self::Failed { error: Some(error) },
        }
    }
}

impl<R: Read> Iterator for IdReader<R> {
    type Item = Result<u64, std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Reading { underlying } => match underlying.read_varint::<u64>() {
                Ok(value) => Some(Ok(value)),
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => None,
                Err(error) => Some(Err(error)),
            },
            Self::Failed { error } => error.take().map(Err),
        }
    }
}

pub fn write_ids_zst_file<
    P: AsRef<Path>,
    E: From<std::io::Error>,
    I: Iterator<Item = Result<u64, E>>,
>(
    output: P,
    values: I,
) -> Result<usize, E> {
    let mut writer = BufWriter::new(Encoder::new(File::create(output)?, 0)?);
    let count = write_ids(&mut writer, values)?;
    writer
        .into_inner()
        .map_err(std::io::Error::from)?
        .finish()?;

    Ok(count)
}

pub fn write_ids<W: Write, E: From<std::io::Error>, I: Iterator<Item = Result<u64, E>>>(
    writer: &mut W,
    values: I,
) -> Result<usize, E> {
    let mut count = 0;

    for result in values {
        let id = result?;
        writer.write_varint(id).map_err(E::from)?;
        count += 1;
    }

    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::{IdReader, write_ids_zst_file};
    use quickcheck::quickcheck;
    use std::sync::LazyLock;

    static TEST_DIR: LazyLock<tempfile::TempDir> = LazyLock::new(|| {
        tempfile::Builder::new()
            .prefix("test-compression")
            .tempdir()
            .unwrap()
    });

    quickcheck! {
        fn round_trip_through_file(values: Vec<u64>) -> bool {
            let path = TEST_DIR.path().join("values.zst");

            write_ids_zst_file::<_, std::io::Error, _>(&path, values.iter().cloned().map(Ok::<_, std::io::Error>)).unwrap();
            let read_values = IdReader::open_zst_file(&path).collect::<Result<Vec<_>, std::io::Error>>().unwrap();

            read_values == values
        }
    }
}
