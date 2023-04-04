# Available experimental data
- X(t) -> Records of sensor glucose
- D -> total carbohydrates ingested at each meal
- U(t) -> insulin delivered through an insulin pump

# Available models
## Dalla Man + CGMS
dG_p = k_p1 - F_cns - (k_1 * G_p) + (k_2 * G_t) - (k_p2 * G_p) - (k_p3 * I_d) + (k_e1 * (k_e2 - G_p)) + ((f * k_abs * Q_gut)/BW)
dG_t = (k_1 * G_p) - (k_2 * G_t) - ((G_t * (V_m0 + (V_mx * X))) / (K_m0 + G_t + (K_mx * X)))
dI_l = (m_2 * I_p) - I_l * (m_1 - ((m_1 * m_6) / (m_6 - 1)))
dI_p = (k_a1 * I_sc1) - (I_p * (m_2 + m_4)) + (k_a2 * I_sc2) + (m_1 * I_l)
dI_1 = -k_i * (I_1 - (I_p / V_I))
dI_d = k_i * (I_1 - I_d)
dQ_sto1 = (-k_gri * Q_sto1) + D(t)δ(t)
dQ_sto2 = (k_gri * Q_sto1) - Q_sto2 * (k_min + ((k_max / 2) - (k_min / 2)) * (tanh(a(Q_sto1 + Q_sto2 - (b * D(t)))) - tanh(c(Q_sto1 + Q_sto2 - (d * D(t)))) + 2))
dQ_gut = Q_sto2 * (k_min + ((k_max / 2) - (k_min / 2)) * (tanh(a(Q_sto1 + Q_sto2 - (b * D(t)))) - tanh(c(Q_sto1 + Q_sto2 - (d * D(t)))) + 2)) - (k_abs * Q_gut)
dX = -p_2U * (I_b - (I_p / V_I)) - (p_2U * X)
dI_sc1 = IIR - (I_sc1 * (k_a1 + kd))
dI_sc2 = (k_d * I_sc1) - (k_a2 * I_sc2)
dSG = (-q_2 * SG) + (CF * m * q_1 * X)

### Parameters to be estimated (identifiable)
k_p2, k_1, k_2, k_p1, k_i, k_e1, k_max, k_min, k_abs, k_p3, k_gri

### Nominal Parameters
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4303268