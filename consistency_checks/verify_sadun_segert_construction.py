import numpy as np
from construct_sadun_segert_ym_solution import fourier_coeff_from_free_params, construct_a3_fourier_coeff

def main():
    # verify_fourier_coeff_from_free_params()
    verify_construct_a3_fourier_coeff()

# Tables from "Stationary Points of the Yang-Mills Action" by Sadun and Segert
# Published in "Communications on Pure and Applied Mathematics" in 1992
# https://doi.org/10.1002/cpa.3160450405 

TABLE_1 = np.array([
    # Cols correspond to l_max = 1, 2, and 3 respectively
    # Rows are the values of the Fourier coefficients of a_3
    [-0.3245945605, -0.3245713584, -0.3245713855],
    [-1.4142100298, -1.4142096286, -1.4142096235],
    [-0.5362985122, -0.5363137348, -0.5363137255], # value for n=2 l=3 is -0.5361317255 in the paper
    [-0.6687388537, -0.6687367686, -0.6687367756], # the true value is:   -0.5363137255
    [ 0.2040423555,  0.2040428056,  0.2040428225], # so they must have made a typo!
    [ 0.0885205177,  0.0885274511,  0.0885274545], 
    [-0.0043693864, -0.0043523523, -0.0043523755],
    [-0.0074408414, -0.0074473554, -0.0074473502],
    [-0.0016187954, -0.0016410762, -0.0016410774],
    [ 0.0020721870,  0.0020742051,  0.0020741999],
    [ 0.0005416187,  0.0005755070,  0.0005755227],
    [-0.0002029798, -0.0001648856, -0.0001648778],
    [            0, -0.0000286352, -0.0000286438],
    [            0, -0.0000394540, -0.0000394588],
    [            0,  0.0000014885,  0.0000014772],
    [            0, -0.0000041032, -0.0000041040],
    [            0,  0.0000016766,  0.0000016566],
    [            0,  0.0000005393,  0.0000005864],
    [            0,             0,  0.0000000454],
    [            0,             0, -0.0000000624],
    [            0,             0, -0.0000000139],
    [            0,             0,  0.0000000130],
    [            0,             0,  0.0000000045],
    [            0,             0, -0.0000000014],
    # Values for S / 8π^2
    [11.4882182679, 11.4882177866, 11.4882177866],
])

TABLE_2 = np.array([
    [3,  3, 5.43281507547525],
    [5,  3, 11.4882177865800],
    [7,  3, 18.9925969277396],
    [5,  5, 21.9519857528157],
    [9,  3, 27.8614519776604],
    [7,  5, 34.1521844389385],
    [11, 3, 38.04236576029],
    [9,  5, 47.96544358790],
    [13, 3, 49.498844315],
    [7,  7, 51.23396533677],
    [15, 3, 62.20367167],
    [11, 5, 63.3120814979],
    [9,  7, 70.1032953685],
    [17, 3, 76.1355939],
    [13, 5, 80.135023031],
    [11, 7, 90.668949991],
    [19, 3, 91.277452],
    [9,  9, 94.14892253],
    [15, 5, 98.3908046],
])

def verify_fourier_coeff_from_free_params():
    free_params = TABLE_1[4:12, 0]
    cons_params = TABLE_1[:4, 0]

    fourier_coeffs = fourier_coeff_from_free_params(free_params, l=1, r=-5, t=3)
    for n in range(4):
        assert abs(fourier_coeffs[n] - cons_params[n]) < 2e-10
    print("fourier_coeff_from_free_params verified with values from Table 1")

def verify_construct_a3_fourier_coeff(match_exact_digits=False):
    r = -5 ; t = 3
    print("Verifying construct_a3_fourier_coeff with values from Table 1")
    for l in range(1, 4):
        computed_coeff = construct_a3_fourier_coeff(l, r, t)
        table_coeff = TABLE_1[:6*l + 6, l - 1]

        if match_exact_digits:
            computed_digits = [f"{value:.9f}" for value in computed_coeff]
            table_digits = [f"{value:.9f}" for value in table_coeff]
            assert computed_digits == table_digits, (computed_digits, table_digits)
        else:
            diffs = [abs(actual - expected) for actual, expected in zip(computed_coeff, table_coeff)]
            id, biggest_error = max(enumerate(diffs), key=lambda x: x[1])
            assert biggest_error < 5e-11, (id, computed_coeff[id], table_coeff[id], diffs[id])
        print(f"verified l={l} col")
    print("construct_a3_fourier_coeff verified with values from Table 1")

if __name__ == "__main__":
    main()
