from cosmosis.datablock import option_section, names
from cosmosis.gaussian_likelihood import GaussianLikelihood
import numpy as np
from scipy.interpolate import interp1d


def get_ratio_from_gammat(gammat1, gammat2, inv_cov):
    # Given two gammats, calculate the ratio
    ones = np.ones(len(gammat1))
    s2 = 1 / (ones @ inv_cov @ ones)
    ratio = s2 * (gammat1 / gammat2) @ inv_cov @ ones
    return float(ratio)



class ShearRatioLikelihood(GaussianLikelihood):
    y_section = "galaxy_shear_xi"
    like_name = "shear_ratio"

    def __init__(self, options):
        super().__init__(options)
        th_limit_low_s14 = options.get_double_array_1d("th_limit_low_s14")
        th_limit_low_s24 = options.get_double_array_1d("th_limit_low_s24")
        th_limit_low_s34 = options.get_double_array_1d("th_limit_low_s34")

        self.theta_low = [th_limit_low_s14, th_limit_low_s24, th_limit_low_s34]
        self.theta_high = options.get_double_array_1d("th_limit_high")
        self.nbin_lens = options.get_int("lens_bins")

        print("logdetC = ", self.log_det_constant)

    def build_data(self):
        filename = self.options["data_file"]
        self.data = np.load(filename, allow_pickle=True, encoding='latin1').item()
        return None, self.data['measured_ratios']


    def build_covariance(self):
        return self.data['ratio_cov']

    def extract_theory_points(self, block):
        radian_to_arcmin = 3437.75
        theta = block[self.y_section, 'theta'] * radian_to_arcmin
        theta_data = self.data['theta_data']
        cov_per_pair = self.data['inv_cov_individual_ratios']

        # If the gamma_t values are already averaged into bins
        # then they should have the same theta values as the
        # data already.  Otherwise we need to interpolated to
        # the correct theta values.
        bin_avg = block.get_bool(self.y_section, 'bin_avg')


        # The source reference bin is the highest redshift bin
        nsource_bins = block[self.y_section, 'nbin_b']
        ref_bin = nsource_bins

        count = 0
        theory_ratios = []

        for li in range(1, self.nbin_lens + 1):
            # Get the reference gamma_t values from the high-z source bin
            # for this lens bin
            gammat_ref = block[self.y_section, f'bin_{li}_{ref_bin}']
            if not bin_avg:
                gammat_ref = interp1d(theta, gammat_ref)(theta_data) 

            # For each source bin (not including the last one, since that is
            # the reference) compute and store the ratio
            for si in range(1, nsource_bins):
                # Remove anything out of the theta range, or with zero reference.
                # This makes this module very fragile - if this changes through
                # the run this will crash.
                mask = (
                    (theta_data > self.theta_low[si - 1][li - 1])
                    & (theta_data <= self.theta_high[li - 1])
                    & (gammat_ref != 0))

                # Extract the gamma_t for this. Again we may need to
                # interpolate to the correct theta values
                gammat_i = block[self.y_section, f'bin_{li}_{si}']
                if not bin_avg:
                    gammat_i = interp1d(theta, gammati)(theta_data) 

                # Either way we have to apply the mask
                gammat_i = gammat_i[mask]

                # Get the chunk of covariance for this one, appropriately masked
                this_cov = cov_per_pair[count][mask].T[mask]

                # Compute the ratio.
                # This is a chunk of our normally-distributed data vector
                ratio = get_ratio_from_gammat(gammat_i, gammat_ref[mask], this_cov)
                theory_ratios.append(ratio)
                count += 1

        return np.array(theory_ratios)


setup, execute, cleanup = ShearRatioLikelihood.build_module()
