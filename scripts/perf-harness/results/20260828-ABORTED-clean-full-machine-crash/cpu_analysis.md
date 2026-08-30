# CPU analysis of `results.csv`

configs: baseline, weighted, weighted+remat, weighted+remat+spectral, eregalloc-c0, eregalloc-c0+spectral, eregalloc-c2, eregalloc-c2+spectral (baseline = `baseline`); targets: 18 (corpus, fixture)

Correctness gate: no miscompiles or failures recorded.

## Runtime per target (median s; Δ vs baseline)

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 0.000979 | 0.000802 (-18.1%) | 0.00073 (-25.4%) | 0.000726 (-25.8%) | 0.000495 (-49.4%) | 0.000844 (-13.8%) | 0.000743 (-24.1%) | 0.000964 (-1.5%) |
| fixture:structs-aarch64 | 0.000734 | 0.00075 (+2.2%) | 0.00071 (-3.3%) | 0.000745 (+1.5%) | 0.000747 (+1.8%) | 0.000832 (+13.4%) | 0.000952 (+29.7%) | 0.000953 (+29.8%) |
| fixture:hello-world-aarch64 | 0.000756 | 0.000871 (+15.2%) | 0.000778 (+2.9%) | 0.000981 (+29.8%) | 0.000978 (+29.4%) | 0.000882 (+16.7%) | 0.001002 (+32.5%) | 0.000743 (-1.7%) |
| fixture:fp-aarch64 | 0.000918 | 0.000925 (+0.8%) | 0.000976 (+6.3%) | 0.000885 (-3.6%) | 0.000831 (-9.5%) | 0.000892 (-2.8%) | 0.001012 (+10.2%) | 0.001013 (+10.3%) |
| fixture:int128-aarch64 | 0.000889 | 0.000942 (+6.0%) | 0.000916 (+3.0%) | 0.001007 (+13.3%) | 0.00102 (+14.7%) | 0.000875 (-1.6%) | 0.000926 (+4.2%) | 0.000906 (+1.9%) |
| fixture:hashmap-aarch64 | 0.000939 | 0.00113 (+20.3%) | 0.001082 (+15.2%) | 0.001045 (+11.3%) | 0.001036 (+10.3%) | 0.001088 (+15.9%) | 0.000874 (-6.9%) | 0.001057 (+12.6%) |
| fixture:itoa-aarch64 | 0.000862 | 0.000902 (+4.6%) | 0.000682 (-20.9%) | 0.000899 (+4.3%) | 0.000903 (+4.8%) | 0.000663 (-23.1%) | 0.000866 (+0.5%) | 0.000893 (+3.6%) |
| fixture:oc-course | 0.000702 | 0.0009 (+28.2%) | 0.000885 (+26.1%) | 0.000913 (+30.1%) | 0.000888 (+26.5%) | 0.000917 (+30.6%) | 0.000913 (+30.1%) | 0.000903 (+28.6%) |
| fixture:stdin-aarch64 | 0.000971 | 0.00095 (-2.2%) | 0.000885 (-8.9%) | 0.000755 (-22.2%) | 0.000938 (-3.4%) | 0.000927 (-4.5%) | 0.000932 (-4.0%) | 0.000773 (-20.4%) |
| fixture:sudoku-aarch64 | 0.001844 | 0.00344 (+86.6%) | 0.003361 (+82.3%) | 0.002772 (+50.3%) | 0.00191 (+3.6%) | 0.001861 (+0.9%) | 0.00197 (+6.8%) | 0.001728 (-6.3%) |
| fixture:sudoku-solver | 0.00091 | 0.00093 (+2.2%) | 0.001028 (+13.0%) | 0.001051 (+15.5%) | 0.000781 (-14.2%) | 0.000782 (-14.1%) | 0.000735 (-19.2%) | 0.000797 (-12.4%) |
| corpus:div_recurrence | 0.6568 | 2.123 (+223.2%) | 2.087 (+217.7%) | 1.837 (+179.7%) | 0.6561 (-0.1%) | 0.6566 (-0.0%) | 0.6562 (-0.1%) | 0.6539 (-0.4%) |
| corpus:dot_product | 3.422 | 12.23 (+257.5%) | 10.9 (+218.5%) | 10.91 (+218.8%) | 3.696 (+8.0%) | 3.689 (+7.8%) | 3.685 (+7.7%) | 3.693 (+7.9%) |
| corpus:elementwise_chain | 5.313 | 16.65 (+213.4%) | 20.11 (+278.5%) | 21.24 (+299.8%) | 7.105 (+33.7%) | 7.031 (+32.3%) | 7.019 (+32.1%) | 7.034 (+32.4%) |
| corpus:gemm_control | 4.648 | 21.47 (+361.9%) | 20.87 (+348.9%) | 21.49 (+362.2%) | 4.584 (-1.4%) | 4.58 (-1.5%) | 4.557 (-2.0%) | 4.603 (-1.0%) |
| corpus:gemm_tiled | 21.27 | 84.22 (+296.0%) | 85.42 (+301.6%) | 85.61 (+302.5%) | 20.2 (-5.0%) | 19.43 (-8.6%) | 19.37 (-8.9%) | 19.42 (-8.7%) |
| corpus:histogram | 4.955 | 16.88 (+240.6%) | 17.28 (+248.8%) | 17.13 (+245.7%) | 4.906 (-1.0%) | 4.899 (-1.1%) | 4.906 (-1.0%) | 4.904 (-1.0%) |
| corpus:jacobi_2d | 2.621 | — | — | — | — | — | — | — |

## Kernel-function static metrics

### kfn_ldr_sp

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| corpus:div_recurrence | 25 | 75 (+200.0%) | 67 (+168.0%) | 54 (+116.0%) | 25 (+0.0%) | 25 (+0.0%) | 25 (+0.0%) | 25 (+0.0%) |
| corpus:dot_product | 44 | 115 (+161.4%) | 93 (+111.4%) | 91 (+106.8%) | 46 (+4.5%) | 46 (+4.5%) | 46 (+4.5%) | 46 (+4.5%) |
| corpus:elementwise_chain | 33 | 102 (+209.1%) | 92 (+178.8%) | 96 (+190.9%) | 33 (+0.0%) | 33 (+0.0%) | 33 (+0.0%) | 33 (+0.0%) |
| corpus:gemm_control | 36 | 81 (+125.0%) | 72 (+100.0%) | 72 (+100.0%) | 37 (+2.8%) | 37 (+2.8%) | 37 (+2.8%) | 37 (+2.8%) |
| corpus:gemm_tiled | 121 | 236 (+95.0%) | 197 (+62.8%) | 197 (+62.8%) | 121 (+0.0%) | 120 (-0.8%) | 121 (+0.0%) | 120 (-0.8%) |
| corpus:histogram | 45 | 105 (+133.3%) | 85 (+88.9%) | 86 (+91.1%) | 48 (+6.7%) | 48 (+6.7%) | 48 (+6.7%) | 48 (+6.7%) |
| corpus:jacobi_2d | 35 | 121 (+245.7%) | — | — | — | — | — | — |

### kfn_str_sp

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| corpus:div_recurrence | 20 | 77 (+285.0%) | 77 (+285.0%) | 63 (+215.0%) | 21 (+5.0%) | 21 (+5.0%) | 22 (+10.0%) | 22 (+10.0%) |
| corpus:dot_product | 27 | 112 (+314.8%) | 112 (+314.8%) | 109 (+303.7%) | 30 (+11.1%) | 30 (+11.1%) | 30 (+11.1%) | 30 (+11.1%) |
| corpus:elementwise_chain | 21 | 100 (+376.2%) | 103 (+390.5%) | 109 (+419.0%) | 23 (+9.5%) | 23 (+9.5%) | 23 (+9.5%) | 23 (+9.5%) |
| corpus:gemm_control | 23 | 81 (+252.2%) | 81 (+252.2%) | 81 (+252.2%) | 26 (+13.0%) | 26 (+13.0%) | 26 (+13.0%) | 26 (+13.0%) |
| corpus:gemm_tiled | 843 | 985 (+16.8%) | 991 (+17.6%) | 991 (+17.6%) | 848 (+0.6%) | 851 (+0.9%) | 848 (+0.6%) | 851 (+0.9%) |
| corpus:histogram | 29 | 99 (+241.4%) | 101 (+248.3%) | 100 (+244.8%) | 33 (+13.8%) | 34 (+17.2%) | 33 (+13.8%) | 34 (+17.2%) |
| corpus:jacobi_2d | 13 | 119 (+815.4%) | — | — | — | — | — | — |

### kfn_mov_imm

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| corpus:div_recurrence | 35 | 35 (+0.0%) | 43 (+22.9%) | 42 (+20.0%) | 35 (+0.0%) | 35 (+0.0%) | 36 (+2.9%) | 36 (+2.9%) |
| corpus:dot_product | 49 | 49 (+0.0%) | 72 (+46.9%) | 71 (+44.9%) | 49 (+0.0%) | 49 (+0.0%) | 49 (+0.0%) | 49 (+0.0%) |
| corpus:elementwise_chain | 143 | 143 (+0.0%) | 156 (+9.1%) | 156 (+9.1%) | 145 (+1.4%) | 145 (+1.4%) | 145 (+1.4%) | 145 (+1.4%) |
| corpus:gemm_control | 32 | 32 (+0.0%) | 41 (+28.1%) | 41 (+28.1%) | 32 (+0.0%) | 32 (+0.0%) | 32 (+0.0%) | 32 (+0.0%) |
| corpus:gemm_tiled | 859 | 859 (+0.0%) | 905 (+5.4%) | 905 (+5.4%) | 861 (+0.2%) | 863 (+0.5%) | 861 (+0.2%) | 863 (+0.5%) |
| corpus:histogram | 54 | 54 (+0.0%) | 77 (+42.6%) | 77 (+42.6%) | 54 (+0.0%) | 55 (+1.9%) | 54 (+0.0%) | 55 (+1.9%) |
| corpus:jacobi_2d | 56 | 56 (+0.0%) | — | — | — | — | — | — |

### kfn_insns

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| corpus:div_recurrence | 132 | 239 (+81.1%) | 239 (+81.1%) | 211 (+59.8%) | 133 (+0.8%) | 133 (+0.8%) | 135 (+2.3%) | 135 (+2.3%) |
| corpus:dot_product | 211 | 367 (+73.9%) | 368 (+74.4%) | 362 (+71.6%) | 216 (+2.4%) | 216 (+2.4%) | 216 (+2.4%) | 216 (+2.4%) |
| corpus:elementwise_chain | 312 | 460 (+47.4%) | 466 (+49.4%) | 476 (+52.6%) | 316 (+1.3%) | 316 (+1.3%) | 316 (+1.3%) | 316 (+1.3%) |
| corpus:gemm_control | 151 | 254 (+68.2%) | 254 (+68.2%) | 254 (+68.2%) | 155 (+2.6%) | 155 (+2.6%) | 155 (+2.6%) | 155 (+2.6%) |
| corpus:gemm_tiled | 2018 | 2277 (+12.8%) | 2290 (+13.5%) | 2290 (+13.5%) | 2025 (+0.3%) | 2029 (+0.5%) | 2025 (+0.3%) | 2029 (+0.5%) |
| corpus:histogram | 226 | 356 (+57.5%) | 361 (+59.7%) | 361 (+59.7%) | 233 (+3.1%) | 235 (+4.0%) | 233 (+3.1%) | 235 (+4.0%) |
| corpus:jacobi_2d | 186 | 378 (+103.2%) | — | — | — | — | — | — |

## Whole-object static metrics

### ldr_sp

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 217 | 237 (+9.2%) | 220 (+1.4%) | 220 (+1.4%) | 208 (-4.1%) | 208 (-4.1%) | 208 (-4.1%) | 208 (-4.1%) |
| fixture:structs-aarch64 | 18 | 17 (-5.6%) | 17 (-5.6%) | 17 (-5.6%) | 17 (-5.6%) | 17 (-5.6%) | 17 (-5.6%) | 17 (-5.6%) |
| fixture:hello-world-aarch64 | 13 | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) |
| fixture:fp-aarch64 | 1666 | 1680 (+0.8%) | 1635 (-1.9%) | 1640 (-1.6%) | 1639 (-1.6%) | 1638 (-1.7%) | 1639 (-1.6%) | 1638 (-1.7%) |
| fixture:int128-aarch64 | 4438 | 4720 (+6.4%) | 4375 (-1.4%) | 4399 (-0.9%) | 4284 (-3.5%) | 4282 (-3.5%) | 4258 (-4.1%) | 4256 (-4.1%) |
| fixture:hashmap-aarch64 | 6960 | 8234 (+18.3%) | 7239 (+4.0%) | 7182 (+3.2%) | 6640 (-4.6%) | 6631 (-4.7%) | 6633 (-4.7%) | 6624 (-4.8%) |
| fixture:itoa-aarch64 | 1018 | 1038 (+2.0%) | 922 (-9.4%) | 922 (-9.4%) | 927 (-8.9%) | 927 (-8.9%) | 927 (-8.9%) | 927 (-8.9%) |
| fixture:oc-course | 25 | 30 (+20.0%) | 25 (+0.0%) | 26 (+4.0%) | 25 (+0.0%) | 25 (+0.0%) | 25 (+0.0%) | 25 (+0.0%) |
| fixture:stdin-aarch64 | 1981 | 3207 (+61.9%) | 2806 (+41.6%) | 2959 (+49.4%) | 1967 (-0.7%) | 1964 (-0.9%) | 1966 (-0.8%) | 1963 (-0.9%) |
| fixture:sudoku-aarch64 | 666 | 891 (+33.8%) | 781 (+17.3%) | 705 (+5.9%) | 649 (-2.6%) | 648 (-2.7%) | 648 (-2.7%) | 647 (-2.9%) |
| fixture:sudoku-solver | 5614 | 6766 (+20.5%) | 6173 (+10.0%) | 6018 (+7.2%) | 5295 (-5.7%) | 5282 (-5.9%) | 5287 (-5.8%) | 5274 (-6.1%) |
| corpus:div_recurrence | 929 | 985 (+6.0%) | 939 (+1.1%) | 943 (+1.5%) | 899 (-3.2%) | 898 (-3.3%) | 896 (-3.6%) | 895 (-3.7%) |
| corpus:dot_product | 1056 | 1150 (+8.9%) | 1085 (+2.7%) | 1113 (+5.4%) | 1030 (-2.5%) | 1029 (-2.6%) | 1027 (-2.7%) | 1026 (-2.8%) |
| corpus:elementwise_chain | 934 | 1009 (+8.0%) | 961 (+2.9%) | 982 (+5.1%) | 904 (-3.2%) | 903 (-3.3%) | 901 (-3.5%) | 900 (-3.6%) |
| corpus:gemm_control | 974 | 1025 (+5.2%) | 976 (+0.2%) | 993 (+2.0%) | 945 (-3.0%) | 944 (-3.1%) | 942 (-3.3%) | 941 (-3.4%) |
| corpus:gemm_tiled | 1028 | 1149 (+11.8%) | 1070 (+4.1%) | 1087 (+5.7%) | 998 (-2.9%) | 996 (-3.1%) | 995 (-3.2%) | 993 (-3.4%) |
| corpus:histogram | 938 | 1000 (+6.6%) | 942 (+0.4%) | 960 (+2.3%) | 911 (-2.9%) | 910 (-3.0%) | 908 (-3.2%) | 907 (-3.3%) |
| corpus:jacobi_2d | 922 | 1010 (+9.5%) | — | — | — | — | — | — |

### str_sp

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 223 | 251 (+12.6%) | 251 (+12.6%) | 251 (+12.6%) | 223 (+0.0%) | 223 (+0.0%) | 223 (+0.0%) | 223 (+0.0%) |
| fixture:structs-aarch64 | 12 | 12 (+0.0%) | 12 (+0.0%) | 12 (+0.0%) | 12 (+0.0%) | 12 (+0.0%) | 12 (+0.0%) | 12 (+0.0%) |
| fixture:hello-world-aarch64 | 13 | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) | 13 (+0.0%) |
| fixture:fp-aarch64 | 1625 | 1658 (+2.0%) | 2048 (+26.0%) | 2052 (+26.3%) | 1633 (+0.5%) | 1632 (+0.4%) | 1633 (+0.5%) | 1632 (+0.4%) |
| fixture:int128-aarch64 | 4422 | 5002 (+13.1%) | 5367 (+21.4%) | 5370 (+21.4%) | 4546 (+2.8%) | 4543 (+2.7%) | 4537 (+2.6%) | 4534 (+2.5%) |
| fixture:hashmap-aarch64 | 6227 | 7917 (+27.1%) | 9687 (+55.6%) | 9554 (+53.4%) | 6355 (+2.1%) | 6349 (+2.0%) | 6348 (+1.9%) | 6342 (+1.8%) |
| fixture:itoa-aarch64 | 700 | 732 (+4.6%) | 740 (+5.7%) | 740 (+5.7%) | 694 (-0.9%) | 694 (-0.9%) | 694 (-0.9%) | 694 (-0.9%) |
| fixture:oc-course | 22 | 28 (+27.3%) | 28 (+27.3%) | 28 (+27.3%) | 23 (+4.5%) | 23 (+4.5%) | 23 (+4.5%) | 23 (+4.5%) |
| fixture:stdin-aarch64 | 1785 | 3155 (+76.8%) | 3394 (+90.1%) | 3501 (+96.1%) | 1866 (+4.5%) | 1807 (+1.2%) | 1865 (+4.5%) | 1806 (+1.2%) |
| fixture:sudoku-aarch64 | 677 | 931 (+37.5%) | 1036 (+53.0%) | 932 (+37.7%) | 685 (+1.2%) | 687 (+1.5%) | 684 (+1.0%) | 686 (+1.3%) |
| fixture:sudoku-solver | 4956 | 6513 (+31.4%) | 7321 (+47.7%) | 7079 (+42.8%) | 5057 (+2.0%) | 4992 (+0.7%) | 5048 (+1.9%) | 4983 (+0.5%) |
| corpus:div_recurrence | 852 | 972 (+14.1%) | 1060 (+24.4%) | 1058 (+24.2%) | 904 (+6.1%) | 864 (+1.4%) | 901 (+5.8%) | 861 (+1.1%) |
| corpus:dot_product | 1214 | 1374 (+13.2%) | 1465 (+20.7%) | 1482 (+22.1%) | 1269 (+4.5%) | 1229 (+1.2%) | 1265 (+4.2%) | 1225 (+0.9%) |
| corpus:elementwise_chain | 850 | 992 (+16.7%) | 1083 (+27.4%) | 1101 (+29.5%) | 903 (+6.2%) | 863 (+1.5%) | 899 (+5.8%) | 859 (+1.1%) |
| corpus:gemm_control | 877 | 998 (+13.8%) | 1086 (+23.8%) | 1098 (+25.2%) | 931 (+6.2%) | 891 (+1.6%) | 927 (+5.7%) | 887 (+1.1%) |
| corpus:gemm_tiled | 1675 | 1880 (+12.2%) | 1974 (+17.9%) | 1986 (+18.6%) | 1731 (+3.3%) | 1694 (+1.1%) | 1727 (+3.1%) | 1690 (+0.9%) |
| corpus:histogram | 1109 | 1234 (+11.3%) | 1324 (+19.4%) | 1335 (+20.4%) | 1160 (+4.6%) | 1121 (+1.1%) | 1156 (+4.2%) | 1117 (+0.7%) |
| corpus:jacobi_2d | 831 | 992 (+19.4%) | — | — | — | — | — | — |

### mov_imm

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 484 | 484 (+0.0%) | 501 (+3.5%) | 501 (+3.5%) | 493 (+1.9%) | 493 (+1.9%) | 493 (+1.9%) | 493 (+1.9%) |
| fixture:structs-aarch64 | 51 | 51 (+0.0%) | 51 (+0.0%) | 51 (+0.0%) | 51 (+0.0%) | 51 (+0.0%) | 51 (+0.0%) | 51 (+0.0%) |
| fixture:hello-world-aarch64 | 8 | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) | 8 (+0.0%) |
| fixture:fp-aarch64 | 1113 | 1113 (+0.0%) | 1554 (+39.6%) | 1555 (+39.7%) | 1133 (+1.8%) | 1134 (+1.9%) | 1133 (+1.8%) | 1134 (+1.9%) |
| fixture:int128-aarch64 | 4073 | 4073 (+0.0%) | 4815 (+18.2%) | 4816 (+18.2%) | 4231 (+3.9%) | 4232 (+3.9%) | 4263 (+4.7%) | 4264 (+4.7%) |
| fixture:hashmap-aarch64 | 6599 | 6599 (+0.0%) | 9517 (+44.2%) | 9478 (+43.6%) | 6817 (+3.3%) | 6829 (+3.5%) | 6811 (+3.2%) | 6823 (+3.4%) |
| fixture:itoa-aarch64 | 700 | 700 (+0.0%) | 823 (+17.6%) | 823 (+17.6%) | 792 (+13.1%) | 792 (+13.1%) | 784 (+12.0%) | 784 (+12.0%) |
| fixture:oc-course | 14 | 14 (+0.0%) | 19 (+35.7%) | 19 (+35.7%) | 14 (+0.0%) | 14 (+0.0%) | 14 (+0.0%) | 14 (+0.0%) |
| fixture:stdin-aarch64 | 2623 | 2623 (+0.0%) | 3329 (+26.9%) | 3334 (+27.1%) | 2704 (+3.1%) | 2654 (+1.2%) | 2699 (+2.9%) | 2650 (+1.0%) |
| fixture:sudoku-aarch64 | 491 | 491 (+0.0%) | 706 (+43.8%) | 690 (+40.5%) | 510 (+3.9%) | 512 (+4.3%) | 507 (+3.3%) | 508 (+3.5%) |
| fixture:sudoku-solver | 4404 | 4404 (+0.0%) | 5877 (+33.4%) | 5842 (+32.7%) | 4561 (+3.6%) | 4513 (+2.5%) | 4543 (+3.2%) | 4496 (+2.1%) |
| corpus:div_recurrence | 786 | 786 (+0.0%) | 965 (+22.8%) | 968 (+23.2%) | 850 (+8.1%) | 810 (+3.1%) | 846 (+7.6%) | 806 (+2.5%) |
| corpus:dot_product | 1135 | 1135 (+0.0%) | 1336 (+17.7%) | 1339 (+18.0%) | 1199 (+5.6%) | 1159 (+2.1%) | 1195 (+5.3%) | 1155 (+1.8%) |
| corpus:elementwise_chain | 897 | 897 (+0.0%) | 1081 (+20.5%) | 1085 (+21.0%) | 963 (+7.4%) | 923 (+2.9%) | 958 (+6.8%) | 918 (+2.3%) |
| corpus:gemm_control | 789 | 789 (+0.0%) | 971 (+23.1%) | 975 (+23.6%) | 853 (+8.1%) | 813 (+3.0%) | 848 (+7.5%) | 808 (+2.4%) |
| corpus:gemm_tiled | 1613 | 1613 (+0.0%) | 1832 (+13.6%) | 1836 (+13.8%) | 1679 (+4.1%) | 1641 (+1.7%) | 1674 (+3.8%) | 1636 (+1.4%) |
| corpus:histogram | 1057 | 1057 (+0.0%) | 1253 (+18.5%) | 1257 (+18.9%) | 1123 (+6.2%) | 1084 (+2.6%) | 1116 (+5.6%) | 1077 (+1.9%) |
| corpus:jacobi_2d | 798 | 798 (+0.0%) | — | — | — | — | — | — |

### insns

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 1640 | 1688 (+2.9%) | 1688 (+2.9%) | 1688 (+2.9%) | 1640 (+0.0%) | 1640 (+0.0%) | 1640 (+0.0%) | 1640 (+0.0%) |
| fixture:structs-aarch64 | 156 | 155 (-0.6%) | 155 (-0.6%) | 155 (-0.6%) | 155 (-0.6%) | 155 (-0.6%) | 155 (-0.6%) | 155 (-0.6%) |
| fixture:hello-world-aarch64 | 85 | 85 (+0.0%) | 85 (+0.0%) | 85 (+0.0%) | 85 (+0.0%) | 85 (+0.0%) | 85 (+0.0%) | 85 (+0.0%) |
| fixture:fp-aarch64 | 7643 | 7690 (+0.6%) | 8478 (+10.9%) | 8488 (+11.1%) | 7644 (+0.0%) | 7643 (+0.0%) | 7644 (+0.0%) | 7643 (+0.0%) |
| fixture:int128-aarch64 | 21824 | 22686 (+3.9%) | 23448 (+7.4%) | 23476 (+7.6%) | 21957 (+0.6%) | 21953 (+0.6%) | 21949 (+0.6%) | 21945 (+0.6%) |
| fixture:hashmap-aarch64 | 32976 | 35942 (+9.0%) | 39637 (+20.2%) | 39408 (+19.5%) | 33006 (+0.1%) | 33003 (+0.1%) | 32982 (+0.0%) | 32979 (+0.0%) |
| fixture:itoa-aarch64 | 4653 | 4705 (+1.1%) | 4720 (+1.4%) | 4720 (+1.4%) | 4656 (+0.1%) | 4656 (+0.1%) | 4640 (-0.3%) | 4640 (-0.3%) |
| fixture:oc-course | 137 | 148 (+8.0%) | 148 (+8.0%) | 149 (+8.8%) | 138 (+0.7%) | 138 (+0.7%) | 138 (+0.7%) | 138 (+0.7%) |
| fixture:stdin-aarch64 | 11192 | 13790 (+23.2%) | 14336 (+28.1%) | 14601 (+30.5%) | 11345 (+1.4%) | 11232 (+0.4%) | 11333 (+1.3%) | 11222 (+0.3%) |
| fixture:sudoku-aarch64 | 3094 | 3573 (+15.5%) | 3783 (+22.3%) | 3587 (+15.9%) | 3107 (+0.4%) | 3111 (+0.5%) | 3099 (+0.2%) | 3101 (+0.2%) |
| fixture:sudoku-solver | 24792 | 27505 (+10.9%) | 29195 (+17.8%) | 28763 (+16.0%) | 24748 (-0.2%) | 24621 (-0.7%) | 24696 (-0.4%) | 24571 (-0.9%) |
| corpus:div_recurrence | 4669 | 4845 (+3.8%) | 5066 (+8.5%) | 5071 (+8.6%) | 4759 (+1.9%) | 4678 (+0.2%) | 4745 (+1.6%) | 4664 (-0.1%) |
| corpus:dot_product | 5743 | 5997 (+4.4%) | 6224 (+8.4%) | 6272 (+9.2%) | 5839 (+1.7%) | 5758 (+0.3%) | 5825 (+1.4%) | 5744 (+0.0%) |
| corpus:elementwise_chain | 4842 | 5059 (+4.5%) | 5286 (+9.2%) | 5329 (+10.1%) | 4935 (+1.9%) | 4854 (+0.2%) | 4919 (+1.6%) | 4838 (-0.1%) |
| corpus:gemm_control | 4756 | 4928 (+3.6%) | 5149 (+8.3%) | 5182 (+9.0%) | 4849 (+2.0%) | 4768 (+0.3%) | 4833 (+1.6%) | 4752 (-0.1%) |
| corpus:gemm_tiled | 6560 | 6888 (+5.0%) | 7122 (+8.6%) | 7155 (+9.1%) | 6656 (+1.5%) | 6579 (+0.3%) | 6640 (+1.2%) | 6563 (+0.0%) |
| corpus:histogram | 5240 | 5427 (+3.6%) | 5655 (+7.9%) | 5688 (+8.5%) | 5334 (+1.8%) | 5255 (+0.3%) | 5316 (+1.5%) | 5237 (-0.1%) |
| corpus:jacobi_2d | 4662 | 4911 (+5.3%) | — | — | — | — | — | — |

### text_bytes

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 6697 | 6889 (+2.9%) | 6889 (+2.9%) | 6889 (+2.9%) | 6697 (+0.0%) | 6697 (+0.0%) | 6697 (+0.0%) | 6697 (+0.0%) |
| fixture:structs-aarch64 | 624 | 620 (-0.6%) | 620 (-0.6%) | 620 (-0.6%) | 620 (-0.6%) | 620 (-0.6%) | 620 (-0.6%) | 620 (-0.6%) |
| fixture:hello-world-aarch64 | 351 | 351 (+0.0%) | 351 (+0.0%) | 351 (+0.0%) | 351 (+0.0%) | 351 (+0.0%) | 351 (+0.0%) | 351 (+0.0%) |
| fixture:fp-aarch64 | 31243 | 31431 (+0.6%) | 34583 (+10.7%) | 34623 (+10.8%) | 31247 (+0.0%) | 31243 (+0.0%) | 31247 (+0.0%) | 31243 (+0.0%) |
| fixture:int128-aarch64 | 89437 | 92885 (+3.9%) | 95933 (+7.3%) | 96045 (+7.4%) | 89969 (+0.6%) | 89953 (+0.6%) | 89937 (+0.6%) | 89921 (+0.5%) |
| fixture:hashmap-aarch64 | 132321 | 144185 (+9.0%) | 158965 (+20.1%) | 158049 (+19.4%) | 132441 (+0.1%) | 132429 (+0.1%) | 132345 (+0.0%) | 132333 (+0.0%) |
| fixture:itoa-aarch64 | 19071 | 19279 (+1.1%) | 19339 (+1.4%) | 19339 (+1.4%) | 19083 (+0.1%) | 19083 (+0.1%) | 19019 (-0.3%) | 19019 (-0.3%) |
| fixture:oc-course | 559 | 603 (+7.9%) | 603 (+7.9%) | 607 (+8.6%) | 563 (+0.7%) | 563 (+0.7%) | 563 (+0.7%) | 563 (+0.7%) |
| fixture:stdin-aarch64 | 45428 | 55820 (+22.9%) | 58004 (+27.7%) | 59064 (+30.0%) | 46040 (+1.3%) | 45588 (+0.4%) | 45992 (+1.2%) | 45548 (+0.3%) |
| fixture:sudoku-aarch64 | 12472 | 14388 (+15.4%) | 15228 (+22.1%) | 14444 (+15.8%) | 12524 (+0.4%) | 12540 (+0.5%) | 12492 (+0.2%) | 12500 (+0.2%) |
| fixture:sudoku-solver | 100724 | 111576 (+10.8%) | 118336 (+17.5%) | 116608 (+15.8%) | 100548 (-0.2%) | 100040 (-0.7%) | 100340 (-0.4%) | 99840 (-0.9%) |
| corpus:div_recurrence | 19147 | 19851 (+3.7%) | 20735 (+8.3%) | 20755 (+8.4%) | 19507 (+1.9%) | 19183 (+0.2%) | 19451 (+1.6%) | 19127 (-0.1%) |
| corpus:dot_product | 23312 | 24328 (+4.4%) | 25236 (+8.3%) | 25428 (+9.1%) | 23696 (+1.6%) | 23372 (+0.3%) | 23640 (+1.4%) | 23316 (+0.0%) |
| corpus:elementwise_chain | 19839 | 20707 (+4.4%) | 21615 (+9.0%) | 21787 (+9.8%) | 20211 (+1.9%) | 19887 (+0.2%) | 20147 (+1.6%) | 19823 (-0.1%) |
| corpus:gemm_control | 19495 | 20183 (+3.5%) | 21067 (+8.1%) | 21199 (+8.7%) | 19867 (+1.9%) | 19543 (+0.2%) | 19803 (+1.6%) | 19479 (-0.1%) |
| corpus:gemm_tiled | 26711 | 28023 (+4.9%) | 28959 (+8.4%) | 29091 (+8.9%) | 27095 (+1.4%) | 26787 (+0.3%) | 27031 (+1.2%) | 26723 (+0.0%) |
| corpus:histogram | 21431 | 22179 (+3.5%) | 23091 (+7.7%) | 23223 (+8.4%) | 21807 (+1.8%) | 21491 (+0.3%) | 21735 (+1.4%) | 21419 (-0.1%) |
| corpus:jacobi_2d | 19119 | 20115 (+5.2%) | — | — | — | — | — | — |

### compile_s

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 0.431 | 0.445 (+3.2%) | 0.431 (+0.0%) | 0.444 (+3.0%) | 0.426 (-1.2%) | 0.435 (+0.9%) | 0.607 (+40.8%) | 0.613 (+42.2%) |
| fixture:structs-aarch64 | 0.126 | 0.148 (+17.5%) | 0.145 (+15.1%) | 0.126 (+0.0%) | 0.131 (+4.0%) | 0.153 (+21.4%) | 0.155 (+23.0%) | 0.145 (+15.1%) |
| fixture:hello-world-aarch64 | 0.133 | 0.145 (+9.0%) | 0.125 (-6.0%) | 0.145 (+9.0%) | 0.145 (+9.0%) | 0.135 (+1.5%) | 0.143 (+7.5%) | 0.152 (+14.3%) |
| fixture:fp-aarch64 | 1.657 | 1.653 (-0.2%) | 1.673 (+1.0%) | 1.675 (+1.1%) | 2.097 (+26.6%) | 2.108 (+27.2%) | 2.597 (+56.7%) | 2.619 (+58.1%) |
| fixture:int128-aarch64 | 5.803 | 5.833 (+0.5%) | 5.841 (+0.7%) | 5.847 (+0.8%) | 6.326 (+9.0%) | 6.313 (+8.8%) | 9.948 (+71.4%) | 9.946 (+71.4%) |
| fixture:hashmap-aarch64 | 8.322 | 8.394 (+0.9%) | 8.435 (+1.4%) | 8.469 (+1.8%) | 9.245 (+11.1%) | 9.258 (+11.2%) | 16.814 (+102.0%) | 16.846 (+102.4%) |
| fixture:itoa-aarch64 | 1.318 | 1.316 (-0.2%) | 1.314 (-0.3%) | 1.319 (+0.1%) | 1.346 (+2.1%) | 1.365 (+3.6%) | 1.769 (+34.2%) | 1.794 (+36.1%) |
| fixture:oc-course | 0.169 | 0.151 (-10.7%) | 0.157 (-7.1%) | 0.166 (-1.8%) | 0.141 (-16.6%) | 0.168 (-0.6%) | 0.176 (+4.1%) | 0.17 (+0.6%) |
| fixture:stdin-aarch64 | 8.256 | 8.307 (+0.6%) | 8.299 (+0.5%) | 8.331 (+0.9%) | 8.392 (+1.6%) | 8.426 (+2.1%) | 9.452 (+14.5%) | 9.416 (+14.1%) |
| fixture:sudoku-aarch64 | 1.324 | 1.319 (-0.4%) | 1.329 (+0.4%) | 1.332 (+0.6%) | 1.371 (+3.5%) | 1.372 (+3.6%) | 1.638 (+23.7%) | 1.632 (+23.3%) |
| fixture:sudoku-solver | 14.612 | 14.699 (+0.6%) | 14.74 (+0.9%) | 14.791 (+1.2%) | 15.393 (+5.3%) | 15.381 (+5.3%) | 18.479 (+26.5%) | 18.535 (+26.8%) |
| corpus:div_recurrence | 5.651 | 5.631 (-0.4%) | 5.707 (+1.0%) | 5.72 (+1.2%) | 5.731 (+1.4%) | 5.718 (+1.2%) | 6.045 (+7.0%) | 5.965 (+5.6%) |
| corpus:dot_product | 5.775 | 5.73 (-0.8%) | 5.8 (+0.4%) | 5.798 (+0.4%) | 5.829 (+0.9%) | 5.818 (+0.7%) | 6.09 (+5.5%) | 6.148 (+6.5%) |
| corpus:elementwise_chain | 5.638 | 5.626 (-0.2%) | 6.126 (+8.7%) | 6.534 (+15.9%) | 6.568 (+16.5%) | 6.539 (+16.0%) | 6.983 (+23.9%) | 6.972 (+23.7%) |
| corpus:gemm_control | 6.523 | 6.457 (-1.0%) | 6.508 (-0.2%) | 6.659 (+2.1%) | 6.526 (+0.0%) | 6.622 (+1.5%) | 6.996 (+7.3%) | 6.966 (+6.8%) |
| corpus:gemm_tiled | 6.897 | 6.573 (-4.7%) | 6.912 (+0.2%) | 6.527 (-5.4%) | 6.834 (-0.9%) | 7.013 (+1.7%) | 6.364 (-7.7%) | 6.314 (-8.5%) |
| corpus:histogram | 6.182 | 6.397 (+3.5%) | 6.169 (-0.2%) | 6.239 (+0.9%) | 7.994 (+29.3%) | 6.303 (+2.0%) | 6.601 (+6.8%) | 6.675 (+8.0%) |
| corpus:jacobi_2d | 6.3 | 6.172 (-2.0%) | — | — | — | — | — | — |

## Aggregate: geometric-mean ratio vs baseline (runtime), by corpus kind

| config | corpus (n) | fixture (n) | all | wins / ties / losses (all) |
|---|---|---|---|---|
| weighted | +262.2% (6) | +10.8% (11) | +68.3% | 2 / 1 / 14 |
| weighted+remat | +266.1% (6) | +5.2% (11) | +63.4% | 4 / 0 / 13 |
| weighted+remat+spectral | +263.1% (6) | +7.3% (11) | +65.0% | 3 / 1 / 13 |
| eregalloc-c0 | +5.0% (6) | -1.3% (11) | +0.9% | 5 / 4 / 8 |
| eregalloc-c0+spectral | +4.1% (6) | +0.5% (11) | +1.7% | 6 / 5 / 6 |
| eregalloc-c2 | +3.9% (6) | +3.8% (11) | +3.8% | 5 / 4 / 8 |
| eregalloc-c2+spectral | +4.1% (6) | +3.0% (11) | +3.4% | 4 / 6 / 7 |

## Paired comparisons (B relative to A; tie = |Δ| < 2% runtime)

### spectral vs uniform (linear, weighted+remat): `weighted+remat+spectral` vs `weighted+remat`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +1.0% | 5 / 4 / 8 | fixture:sudoku-aarch64 -17.5% | fixture:itoa-aarch64 +31.8% | YES (drop fixture:itoa-aarch64 → -0.7%) |
| kfn_ldr_sp | -3.0% | 2 / 2 / 2 | corpus:div_recurrence -19.4% | corpus:elementwise_chain +4.3% | YES (drop corpus:div_recurrence → +0.7%) |
| kfn_str_sp | -3.0% | 3 / 2 / 1 | corpus:div_recurrence -18.2% | corpus:elementwise_chain +5.8% | YES (drop corpus:div_recurrence → +0.4%) |
| kfn_mov_imm | -0.6% | 2 / 4 / 0 | corpus:div_recurrence -2.3% | corpus:elementwise_chain +0.0% | YES (drop corpus:div_recurrence → -0.3%) |
| kfn_insns | -2.0% | 2 / 3 / 1 | corpus:div_recurrence -11.7% | corpus:elementwise_chain +2.1% | YES (drop corpus:div_recurrence → +0.1%) |
| compile_s | +0.9% | 3 / 0 / 14 | fixture:structs-aarch64 -13.1% | fixture:hello-world-aarch64 +16.0% | YES (drop fixture:hello-world-aarch64 → +0.0%) |

### spectral vs uniform (eregalloc c0): `eregalloc-c0+spectral` vs `eregalloc-c0`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +0.8% | 5 / 7 / 5 | fixture:itoa-aarch64 -26.6% | fixture:pure-rust-aarch64 +70.5% | YES (drop fixture:pure-rust-aarch64 → -2.4%) |
| kfn_ldr_sp | -0.1% | 1 / 5 / 0 | corpus:gemm_tiled -0.8% | corpus:div_recurrence +0.0% | YES (drop corpus:gemm_tiled → +0.0%) |
| kfn_str_sp | +0.6% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +3.0% | YES (drop corpus:histogram → +0.1%) |
| kfn_mov_imm | +0.3% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +1.9% | YES (drop corpus:histogram → +0.0%) |
| kfn_insns | +0.2% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +0.9% | YES (drop corpus:histogram → +0.0%) |
| compile_s | +0.6% | 7 / 0 / 10 | corpus:histogram -21.2% | fixture:oc-course +19.1% | YES (drop fixture:oc-course → -0.5%) |

### spectral vs uniform (eregalloc c2): `eregalloc-c2+spectral` vs `eregalloc-c2`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | -0.4% | 4 / 9 / 4 | fixture:hello-world-aarch64 -25.8% | fixture:pure-rust-aarch64 +29.7% | YES (drop fixture:hello-world-aarch64 → +1.4%) |
| kfn_ldr_sp | -0.1% | 1 / 5 / 0 | corpus:gemm_tiled -0.8% | corpus:div_recurrence +0.0% | YES (drop corpus:gemm_tiled → +0.0%) |
| kfn_str_sp | +0.6% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +3.0% | YES (drop corpus:histogram → +0.1%) |
| kfn_mov_imm | +0.3% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +1.9% | YES (drop corpus:histogram → +0.0%) |
| kfn_insns | +0.2% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:histogram +0.9% | YES (drop corpus:histogram → +0.0%) |
| compile_s | -0.1% | 9 / 0 / 8 | fixture:structs-aarch64 -6.5% | fixture:hello-world-aarch64 +6.3% | YES (drop fixture:structs-aarch64 → +0.3%) |

### eregalloc c0 vs linear baseline: `eregalloc-c0` vs `baseline`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +0.9% | 5 / 4 / 8 | fixture:pure-rust-aarch64 -49.4% | corpus:elementwise_chain +33.7% | YES (drop corpus:elementwise_chain → -0.9%) |
| kfn_ldr_sp | +2.3% | 0 / 3 / 3 | corpus:div_recurrence +0.0% | corpus:histogram +6.7% | no |
| kfn_str_sp | +8.7% | 0 / 0 / 6 | corpus:gemm_tiled +0.6% | corpus:histogram +13.8% | no |
| kfn_mov_imm | +0.3% | 0 / 4 / 2 | corpus:div_recurrence +0.0% | corpus:elementwise_chain +1.4% | YES (drop corpus:elementwise_chain → +0.0%) |
| kfn_insns | +1.7% | 0 / 0 / 6 | corpus:gemm_tiled +0.3% | corpus:histogram +3.1% | no |
| compile_s | +5.5% | 3 / 0 / 14 | fixture:oc-course -16.6% | corpus:histogram +29.3% | no |

### eregalloc c2 vs linear baseline: `eregalloc-c2` vs `baseline`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +3.8% | 5 / 4 / 8 | fixture:pure-rust-aarch64 -24.1% | fixture:hello-world-aarch64 +32.5% | no |
| kfn_ldr_sp | +2.3% | 0 / 3 / 3 | corpus:div_recurrence +0.0% | corpus:histogram +6.7% | no |
| kfn_str_sp | +9.6% | 0 / 0 / 6 | corpus:gemm_tiled +0.6% | corpus:histogram +13.8% | no |
| kfn_mov_imm | +0.7% | 0 / 3 / 3 | corpus:dot_product +0.0% | corpus:div_recurrence +2.9% | YES (drop corpus:div_recurrence → +0.3%) |
| kfn_insns | +2.0% | 0 / 0 / 6 | corpus:gemm_tiled +0.3% | corpus:histogram +3.1% | no |
| compile_s | +23.7% | 1 / 0 / 16 | corpus:gemm_tiled -7.7% | fixture:hashmap-aarch64 +102.0% | no |

### eregalloc c2 vs c0: `eregalloc-c2` vs `eregalloc-c0`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +2.9% | 5 / 6 / 6 | fixture:hashmap-aarch64 -15.6% | fixture:pure-rust-aarch64 +50.1% | YES (drop fixture:pure-rust-aarch64 → +0.5%) |
| kfn_ldr_sp | +0.0% | 0 / 6 / 0 | corpus:div_recurrence +0.0% | corpus:div_recurrence +0.0% | YES (drop corpus:div_recurrence → +0.0%) |
| kfn_str_sp | +0.8% | 0 / 5 / 1 | corpus:dot_product +0.0% | corpus:div_recurrence +4.8% | YES (drop corpus:div_recurrence → +0.0%) |
| kfn_mov_imm | +0.5% | 0 / 5 / 1 | corpus:dot_product +0.0% | corpus:div_recurrence +2.9% | YES (drop corpus:div_recurrence → +0.0%) |
| kfn_insns | +0.2% | 0 / 5 / 1 | corpus:dot_product +0.0% | corpus:div_recurrence +1.5% | YES (drop corpus:div_recurrence → +0.0%) |
| compile_s | +17.3% | 3 / 0 / 14 | corpus:histogram -17.4% | fixture:hashmap-aarch64 +81.9% | no |

### weighted vs baseline (linear): `weighted` vs `baseline`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | +68.3% | 2 / 1 / 14 | fixture:pure-rust-aarch64 -18.1% | corpus:gemm_control +361.9% | no |
| kfn_ldr_sp | +162.5% | 0 / 0 / 7 | corpus:gemm_tiled +95.0% | corpus:jacobi_2d +245.7% | no |
| kfn_str_sp | +271.6% | 0 / 0 / 7 | corpus:gemm_tiled +16.8% | corpus:jacobi_2d +815.4% | no |
| kfn_mov_imm | +0.0% | 0 / 7 / 0 | corpus:div_recurrence +0.0% | corpus:div_recurrence +0.0% | YES (drop corpus:div_recurrence → +0.0%) |
| kfn_insns | +61.1% | 0 / 0 / 7 | corpus:gemm_tiled +12.8% | corpus:jacobi_2d +103.2% | no |
| compile_s | +0.7% | 10 / 0 / 8 | fixture:oc-course -10.7% | fixture:structs-aarch64 +17.5% | YES (drop fixture:structs-aarch64 → -0.2%) |

### weighted+remat vs weighted (linear): `weighted+remat` vs `weighted`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | -2.9% | 10 / 3 / 4 | fixture:itoa-aarch64 -24.4% | corpus:elementwise_chain +20.8% | YES (drop fixture:itoa-aarch64 → -1.4%) |
| kfn_ldr_sp | -14.5% | 6 / 0 / 0 | corpus:dot_product -19.1% | corpus:elementwise_chain -9.8% | no |
| kfn_str_sp | +0.9% | 0 / 3 / 3 | corpus:div_recurrence +0.0% | corpus:elementwise_chain +3.0% | no |
| kfn_mov_imm | +24.9% | 0 / 0 / 6 | corpus:gemm_tiled +5.4% | corpus:dot_product +46.9% | no |
| kfn_insns | +0.6% | 0 / 2 / 4 | corpus:div_recurrence +0.0% | corpus:histogram +1.4% | no |
| compile_s | -0.0% | 6 / 0 / 11 | fixture:hello-world-aarch64 -13.8% | corpus:elementwise_chain +8.9% | YES (drop fixture:hello-world-aarch64 → +0.9%) |

### eregalloc c0 vs linear weighted+remat: `eregalloc-c0` vs `weighted+remat`

| metric | geomean Δ | wins / ties / losses | best (target, Δ) | worst (target, Δ) | single-target-driven? |
|---|---|---|---|---|---|
| run_median_s | -38.3% | 11 / 1 / 5 | corpus:gemm_control -78.0% | fixture:itoa-aarch64 +32.4% | no |
| kfn_ldr_sp | -52.3% | 6 / 0 / 0 | corpus:elementwise_chain -64.1% | corpus:gemm_tiled -38.6% | no |
| kfn_str_sp | -66.3% | 6 / 0 / 0 | corpus:elementwise_chain -77.7% | corpus:gemm_tiled -14.4% | no |
| kfn_mov_imm | -19.7% | 6 / 0 / 0 | corpus:dot_product -31.9% | corpus:gemm_tiled -4.9% | no |
| kfn_insns | -34.8% | 6 / 0 / 0 | corpus:div_recurrence -44.4% | corpus:gemm_tiled -11.6% | no |
| compile_s | +4.6% | 4 / 0 / 13 | fixture:oc-course -10.2% | corpus:histogram +29.6% | no |

## Compile time (s) per target and config

| target | baseline | weighted | weighted+remat | weighted+remat+spectral | eregalloc-c0 | eregalloc-c0+spectral | eregalloc-c2 | eregalloc-c2+spectral |
|---|---|---|---|---|---|---|---|---|
| fixture:pure-rust-aarch64 | 0.431 | 0.445 | 0.431 | 0.444 | 0.426 | 0.435 | 0.607 | 0.613 |
| fixture:structs-aarch64 | 0.126 | 0.148 | 0.145 | 0.126 | 0.131 | 0.153 | 0.155 | 0.145 |
| fixture:hello-world-aarch64 | 0.133 | 0.145 | 0.125 | 0.145 | 0.145 | 0.135 | 0.143 | 0.152 |
| fixture:fp-aarch64 | 1.657 | 1.653 | 1.673 | 1.675 | 2.097 | 2.108 | 2.597 | 2.619 |
| fixture:int128-aarch64 | 5.803 | 5.833 | 5.841 | 5.847 | 6.326 | 6.313 | 9.948 | 9.946 |
| fixture:hashmap-aarch64 | 8.322 | 8.394 | 8.435 | 8.469 | 9.245 | 9.258 | 16.81 | 16.85 |
| fixture:itoa-aarch64 | 1.318 | 1.316 | 1.314 | 1.319 | 1.346 | 1.365 | 1.769 | 1.794 |
| fixture:oc-course | 0.169 | 0.151 | 0.157 | 0.166 | 0.141 | 0.168 | 0.176 | 0.17 |
| fixture:stdin-aarch64 | 8.256 | 8.307 | 8.299 | 8.331 | 8.392 | 8.426 | 9.452 | 9.416 |
| fixture:sudoku-aarch64 | 1.324 | 1.319 | 1.329 | 1.332 | 1.371 | 1.372 | 1.638 | 1.632 |
| fixture:sudoku-solver | 14.61 | 14.7 | 14.74 | 14.79 | 15.39 | 15.38 | 18.48 | 18.54 |
| corpus:div_recurrence | 5.651 | 5.631 | 5.707 | 5.72 | 5.731 | 5.718 | 6.045 | 5.965 |
| corpus:dot_product | 5.775 | 5.73 | 5.8 | 5.798 | 5.829 | 5.818 | 6.09 | 6.148 |
| corpus:elementwise_chain | 5.638 | 5.626 | 6.126 | 6.534 | 6.568 | 6.539 | 6.983 | 6.972 |
| corpus:gemm_control | 6.523 | 6.457 | 6.508 | 6.659 | 6.526 | 6.622 | 6.996 | 6.966 |
| corpus:gemm_tiled | 6.897 | 6.573 | 6.912 | 6.527 | 6.834 | 7.013 | 6.364 | 6.314 |
| corpus:histogram | 6.182 | 6.397 | 6.169 | 6.239 | 7.994 | 6.303 | 6.601 | 6.675 |
| corpus:jacobi_2d | 6.3 | 6.172 | — | — | — | — | — | — |
