use statrs::statistics::Data;
use statrs::statistics::OrderStatistics;

pub fn arange(start: f64, end: f64, step: f64) -> Vec<f64> {
    let mut range = Vec::new();
    let mut location = start;

    while location < end {
        range.push(location);
        location += step;
    }
    range
}

pub fn histogram(data: Vec<f64>, bins: Vec<f64>) -> Vec<i32> {
    let mut counts = vec![0; bins.len() - 1];

    for x in data {
        // find the bin index
        if let Some(idx) = bins.windows(2).position(|w| x >= w[0] && x < w[1]) {
            counts[idx] += 1;
        } else if x == *bins.last().unwrap() {
            // include right edge like NumPy does
            counts[bins.len() - 2] += 1;
        }
    }

    counts
}

/// Calculates the Freedman–Diaconis_rule for binwidth.
pub fn calculate_fd(data: Vec<f64>) -> f64 {
    let mut x = Data::new(data.clone());
    let iqr = x.upper_quartile() - x.lower_quartile();
    let n = data.clone().len() as f64;
    2. * (iqr / n.powf(1. / 3.))
}

/// Histogram that automatically selects the "optimal" bins using the Freedman-Diaconis rule
pub fn histogram_fd(data: Vec<f64>, end: f64) -> Vec<usize> {
    let bin_width = calculate_fd(data.clone());
    let bins = arange(0., end, bin_width);
    histogram(data, bins)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::iter::zip;

    #[test]
    fn test_histogram() {
        // testing against numpy histogram
        let data = arange(0., 100., 1.);
        let bins = arange(10., 20., 0.5);
        let result = histogram(data, bins);
        let answer = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        for (r, a) in zip(result, answer) {
            assert_eq!(r, a as usize);
        }
    }

    #[test]
    fn test_arange() {
        //testing against np.arange
        let result = arange(0., 10., 1.);
        let answer = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9.];
        for (r, a) in zip(result, answer) {
            assert_eq!(r, a)
        }

        let result = arange(0., 0.5, 0.1);
        let answer = [0., 0.1, 0.2, 0.3, 0.4];
        for (r, a) in zip(result, answer) {
            assert!((r - a).abs() < 1e-7)
        }
    }

    #[test]
    fn test_fd() {
        let data = arange(0., 100., 1.);
        let result = calculate_fd(data);
        let answer = 21.616;
        assert!((result - answer).abs() < 1e-3)
    }
}
