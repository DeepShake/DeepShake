import gdown

urls = [
    "https://drive.google.com/uc?id=11-WfN2RQI97muI9Nh0_d2lBovQ_zcBeE",
    "https://drive.google.com/uc?id=12iGqVYJrzhEFuoUb6wMib5cerQ6pnORi",
    "https://drive.google.com/uc?id=14g1Xt9yiL8oNCwcJxQ8ulIadH37u7eDW",
    "https://drive.google.com/uc?id=17Snwlm5naETzvD5v2iMMYjXL8oXkkwil",
    "https://drive.google.com/uc?id=11l9LLDSwnnhdaSgzsAllMxKiLa0HIqwq",
    "https://drive.google.com/uc?id=15g_dPpsJw7rV_OcuD4SpCkqW_K5PSQX-",
    "https://drive.google.com/uc?id=1-IbDqSE5zsrzngNZ_7PScVOsM_UfCYQ6",
    "https://drive.google.com/uc?id=15NtUAcFwE3xFEdp-pJ8Q0uYPWRR378wk",
    "https://drive.google.com/uc?id=1-OgsyIJABKrlivEkf8dJnujWGfrRWa-n",
    "https://drive.google.com/uc?id=112bkb6ifP5SRt_h1HseY_nbPnC4lV6tq",
    "https://drive.google.com/uc?id=12e6nB3Z_nMKqYNGwDXgjO58KRcisn2KK",
    "https://drive.google.com/uc?id=1pQXSNalJz_xSTZgAddWf7_m4plto-3Co",
    "https://drive.google.com/uc?id=119KTBeuxd8RNutBrIP3jAM65mHEAyQmf",
    "https://drive.google.com/uc?id=1-JlOWjWeVspgO0Vge1idsvf92QEM4pLQ",
    "https://drive.google.com/uc?id=1-UTJSumkB0D7X_6P-81cZ63kNdrMAJAl",
    "https://drive.google.com/uc?id=16y1QHfBQEuw3bJoIICycxrF-e05OfXJr",
    "https://drive.google.com/uc?id=13zd8uIO-dq-Pt_v9YBmCVLkZ3t9gAkcB",
    "https://drive.google.com/uc?id=15b9IGLxKy5wh3KPDdfdJajwPiJggLku7",
    "https://drive.google.com/uc?id=15ILJ60eD4S49n_4xi9OZ_abu_a-eQInj",
    "https://drive.google.com/uc?id=10uabAqv7QK08UjdI0AmqVjEkoVvifNlS",
    "https://drive.google.com/uc?id=10v3sGPjX6tYjJrL6TlwVRLbH5mNpIk1B",
    "https://drive.google.com/uc?id=13RdhDTWfSd8cqEHEEeQ0LB93qitICV4o",
    "https://drive.google.com/uc?id=1Le_mZmwnlO2nxZN17B_Tpje46CR07WWF",
    "https://drive.google.com/uc?id=11XQF7yQ2-xrAF7Kiiop58QPfKdGv3cmz",
    "https://drive.google.com/uc?id=1eRh0kN1MBX8nt-hRA7jk0t1KYWHMkfr3",
    "https://drive.google.com/uc?id=17RkhvvkZwiuKLNs0l1HXldJr8GFyM35W",
    "https://drive.google.com/uc?id=17Q3NwRrNioEz-e_appeoCqdSiPJxbZTT",
    "https://drive.google.com/uc?id=12-5W3pxvgpRpN0r-rnGHP4bgdqjjpCvy",
    "https://drive.google.com/uc?id=16Lp0jZx2swG-hFYxqUZLonZbUGdfqh4Q",
    "https://drive.google.com/uc?id=13NPZEYU6IjHde0jF4LQQc1BCym8QE-T4",
    "https://drive.google.com/uc?id=17BhH4CdgzO5t5iqbCN7sX3vfTvyiqumm",
    "https://drive.google.com/uc?id=12XaEvCEHn014eLQCb_xEh2sRGcUT0IPl",
    "https://drive.google.com/uc?id=15HV2S3bUPNZZA0xhBXy-phnTn8brO4QM",
    "https://drive.google.com/uc?id=11CvdGpXfri8-gCS-R9VktCbm5Yae-KV6",
    "https://drive.google.com/uc?id=176AQnhzrvbzS2LvBuIrrJGeO1cm-yAgj",
    "https://drive.google.com/uc?id=16ma8lkZ28qptZ3uBdYgBtwZ_Vbpa9L4N",
    "https://drive.google.com/uc?id=16o5jTnwHq4a3vrsoDAyPKphFhqCZ3xQu",
    "https://drive.google.com/uc?id=17QQk57ax4hqdvW7itHFTqwGHBXQ_oP7d",
    "https://drive.google.com/uc?id=134CEVBuzZO0i6Cymuta6H-T96tvuWWLG",
    "https://drive.google.com/uc?id=165QHOp5iul3Gj0tG6UObp8E2qI_t2CW_",
    "https://drive.google.com/uc?id=12fXYZZJXwthWfa25GE8cF_gBNQbWki1H",
    "https://drive.google.com/uc?id=163bSOAL1CCQOSCHMX4rXwa58Bs_b40oL",
    "https://drive.google.com/uc?id=12r3L5Yx7--hJHOzL4ryZWFxnMry1zgth",
    "https://drive.google.com/uc?id=1-Gt_8ysqHX27DL2xWZX7ifudmLX1YaDm",
    "https://drive.google.com/uc?id=17IWS-20Poe9M_Mte61znsmSMd1dzr7iL",
    "https://drive.google.com/uc?id=1-FThmc_qpG3KG5Jv5_yLMBEsqQxX367M",
    "https://drive.google.com/uc?id=14_6dnOL5ZkE0guxGx7VI9axsBXQLtDzH",
    "https://drive.google.com/uc?id=150Zli60OrmHnl97BPxY2m3qGa2ImLJVa",
    "https://drive.google.com/uc?id=167Q9KYXlfdblLhGs057WHgUt30daE-Y3",
    "https://drive.google.com/uc?id=1-fBDdQOPfYSeWnrcuGk7-oIYXgzT1Ve-",
    "https://drive.google.com/uc?id=1-24p0eUljeOY1_J4ymX-bH0Qm4IFqlnT",
    "https://drive.google.com/uc?id=172Am0D_pcQXE2LMN9laJvMZ_rS7bEFDy",
    "https://drive.google.com/uc?id=13lTW1UJWaoEfFv9gYrV9XFn8esY39z-X",
    "https://drive.google.com/uc?id=1gLuPSl55JAmqD4cuprafgOeDgiMwm2JS",
    "https://drive.google.com/uc?id=15Nlr0qcbg9ov44Nj9Jqza7ugHy0IubLr",
    "https://drive.google.com/uc?id=12dUqOY7qTlY0msJIlUE04WSbPVI_8Ccd",
    "https://drive.google.com/uc?id=11gRFBKQLwblpyZDLy8pP5feSy38yqAKv",
    "https://drive.google.com/uc?id=12YNrk5Z1L-gVpbjUgOu0Up3BvDU-9IRR",
    "https://drive.google.com/uc?id=134rtK6iB417EEv5vSBSsQqi_Sabnqgp4",
    "https://drive.google.com/uc?id=17V22UgJxaOrWBEatCovEF7y1wS8F03HT",
    "https://drive.google.com/uc?id=11BLPQZ-dbWRSuhCmMnRNsvJu1xRUS78l",
    "https://drive.google.com/uc?id=17fBA2udS9UHGOreTkFl7GVkTMNYpQOzN",
    "https://drive.google.com/uc?id=10kywnyRlfs6n0ib9xe9I8J9eupecTgtp",
    "https://drive.google.com/uc?id=12zpJ7V4W6Gyx1hEkTioxrDPzCbL69XqO",
    "https://drive.google.com/uc?id=13UcGn1_RnosrLYyhmbCJ4w73MjpySNUB",
    "https://drive.google.com/uc?id=14X5C1_x58_hLVE6PeaLyi6Aw-uNo8Bwh",
    "https://drive.google.com/uc?id=1-ekyf2EmNEtJqm-h-T7SzRJFWDEjsvss",
    "https://drive.google.com/uc?id=14POBLq1wRwnnaIfq9Jspy2WBonrdIzp_",
    "https://drive.google.com/uc?id=14KBA28v7HjpUK-kO4EvDAKrcfkbf4UYd",
    "https://drive.google.com/uc?id=1698jILLWP3AWTIhDe1H3Lwu-8EyIRK0z",
    "https://drive.google.com/uc?id=11LD8RPn1v_l61BahPDtiZ5R0RQToUyUo",
    "https://drive.google.com/uc?id=14AxtTS9MLZ52hEkpw-DVSnR6mCLIOaAQ",
    "https://drive.google.com/uc?id=14aBKGk-MKmVujUnmGwCykEYyBqPNJW7H",
    "https://drive.google.com/uc?id=13Z1i58brsLyEcdMQOlZkowe1CVQyQE1d",
    "https://drive.google.com/uc?id=12C2gEst8wufEwM5y54jm6CD-0SK4hxsT",
    "https://drive.google.com/uc?id=14mJ7ItmcPHiOHJGGRbZ4xnjr5vi_D3mq",
    "https://drive.google.com/uc?id=1UnuSB9Eux72KbmGnB5Di8-oabZ30huG9",
    "https://drive.google.com/uc?id=12PJynoPNrGBGYSHqnbqwZHve33Ryj8jW",
    "https://drive.google.com/uc?id=1-f_R9orGTswaxfYi8c6LNNsVlI8DLVr1",
    "https://drive.google.com/uc?id=16lReTyxypYJjTnabBDHyPTQ2-4BBynP3",
    "https://drive.google.com/uc?id=13tEaoT54dv_z6y2whkBD4NIZONsQxM8a",
    "https://drive.google.com/uc?id=11ROJXaZCcXOAZhylatv7ef5Ri4_XpahM",
    "https://drive.google.com/uc?id=15db46uZGz18-ePpKjsYRzz93qeqIAs3B",
    "https://drive.google.com/uc?id=1217LkD6MEMxiOYdfJPEhlv2cuPOg5app",
    "https://drive.google.com/uc?id=1pbJvbZPWtholCsapiVvqVLcDr7KJBukl",
    "https://drive.google.com/uc?id=1F-6DnmdHxEiskdmAtg9pMe_mTcE7bOav",
    "https://drive.google.com/uc?id=1-Uqzqg0zDrR85wveIDVjlaBBmuFlauHs",
    "https://drive.google.com/uc?id=14npxNMwqIy9XvuoDtXWkNrjPdbuNfA_9",
    "https://drive.google.com/uc?id=17N2xq-JNcrZxis2Yr6-C9Z3dyAHeJCRe",
    "https://drive.google.com/uc?id=1-H3F0ON0BHlo5KiYAkrBcPysYu9C7PcM",
    "https://drive.google.com/uc?id=14C86BNbOLndZZ-nKB7P7-wsOcRz3qzyr",
    "https://drive.google.com/uc?id=16LhT1bxe3Io9wJGXo3uwq3tY7pvkISuT",
    "https://drive.google.com/uc?id=11iRFEUJFpk1AtxxZpn376F3G0RSMkTbh",
    "https://drive.google.com/uc?id=16k8rtzdjhxqnmWqlIuZ_APY5XpNsXmgO",
    "https://drive.google.com/uc?id=13P5mPS5-4XTlD03XSvX9jnYIDy9-KISs",
    "https://drive.google.com/uc?id=15Vau_66XjczFQLVrqZyJovwOO2s-n53J",
    "https://drive.google.com/uc?id=16gfSls0yDMZRPXfEBp8iLNqzrA35Q5xs",
    "https://drive.google.com/uc?id=13fkFMEVQzrf_LteE7yudzQvIXr7IWGf_",
    "https://drive.google.com/uc?id=11nBZHbOrq-aPaQLO-8brvQBS8UNajEtv",
    "https://drive.google.com/uc?id=14fkFjnXIXUX9UrRYFe6s4MmcLWNvnNy4",
    "https://drive.google.com/uc?id=12MdrGGtCd8urUXS01GcqN4qSq-l9yeoJ",
    "https://drive.google.com/uc?id=17WOPidqVDEJ2VQfubshPjYq8aho5U_p6",
    "https://drive.google.com/uc?id=11HXZi5rOtvurXKV2t1z1h5yQ_k2YJ4NF",
    "https://drive.google.com/uc?id=13jetIJRuBr_lE_AeC7n_ccb40gnwiBHK",
    "https://drive.google.com/uc?id=112w5BGvvnBHtVoXAw5l7ALC56-Kj6HEZ",
    "https://drive.google.com/uc?id=11tDqUq_0_1jtpDv8I0tHo8B2yHdIgYyT",
    "https://drive.google.com/uc?id=1-DXbvcWin44_kv-s0ZAGzb7PPvCrc8it",
    "https://drive.google.com/uc?id=1-NNtmKoh131iy_ULV_o2ARiz8KTC8Uvu",
    "https://drive.google.com/uc?id=12v0lQsL_mDVPExOZYHT_wnvf0UZcaxXL",
    "https://drive.google.com/uc?id=14T9_oJZKlkIP0qeM-ILqu3VBqJrptJps",
    "https://drive.google.com/uc?id=12lZ3Ypcxe99qIF-_2D7VmDKG2-7592Zf",
    "https://drive.google.com/uc?id=10wBH54d_P_z5diMLIgQrPfjAMPPSReUu",
    "https://drive.google.com/uc?id=13anE4gDzSIfn4T6H-eZFOl0BZZ2tfKfH",
    "https://drive.google.com/uc?id=15SP1QSpE7DgQ29CIU4b39Z1ugnv9ylvY",
    "https://drive.google.com/uc?id=16ibIBnOSn5vVEy2HJLqnvntlkn6k_zY5",
    "https://drive.google.com/uc?id=15PGMMZM15o98Ct6aJQ1xDEB0OjL9vbiJ",
    "https://drive.google.com/uc?id=15PsBH7ti6VgZ8oKAsxPkEGWHx7BGLVld",
    "https://drive.google.com/uc?id=154_UAtlw4e2FSj-GR_GPNd9XmJKEwkb_",
    "https://drive.google.com/uc?id=13G3QMOQSYnPJi2aOqlQqMDm5jsjuRPMO",
    "https://drive.google.com/uc?id=15Yf9v5ynSiCLtIaglYcGfeiCtLAPdLmr",
    "https://drive.google.com/uc?id=12saNBzG4tWrM53rE445j0s4X5RNAsFLc",
    "https://drive.google.com/uc?id=14u1rJN7LeJ3B0pzth9wC_eyBbZLj9mVt",
    "https://drive.google.com/uc?id=12C1vplLMZHW89k-E3sKSdtTsof2m6OyR",
    "https://drive.google.com/uc?id=14t5kC3JYBOn0el8IxtWcD-LgQ1KS8u5o",
]

for url in urls:
    gdown.download(url)