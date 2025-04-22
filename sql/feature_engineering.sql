-- Drop old view if exists
DROP VIEW IF EXISTS v_credit_features;

-- Create a polished view with all code mappings and engineered features
CREATE VIEW v_credit_features AS
SELECT
  -- Raw fields
  Status_of_existing_checking_account,
  Duration_in_month,
  Credit_history,
  Purpose,
  Credit_amount,
  Savings_account_bonds,
  Present_employment_since,
  Installment_rate_in_percentage_of_disposable_income,
  Personal_status_and_sex,
  Other_debtors_guarantors,
  Present_residence_since,
  Property,
  Age_in_years,
  Other_installment_plans,
  Housing,
  Number_of_existing_credits_at_this_bank,
  Job,
  Number_of_people_being_liable_to_provide_maintenance_for,
  Telephone,
  Foreign_worker,
  Credit_risk,

  -- Engineered numeric feature
  CAST(Credit_amount AS REAL) / NULLIF(Duration_in_month,0) AS credit_per_month,


      -- Age groups as categorical
  CASE
    WHEN Age_in_years < 25 THEN 'Under_25'
    WHEN Age_in_years BETWEEN 25 AND 40 THEN '25_to_40'
    ELSE 'Over_40'
 END AS age_group,

    -- One-hot encoded columns for age groups
 CAST(Age_in_years < 25 AS INTEGER) AS age_under_25,
 CAST(Age_in_years BETWEEN 25 AND 40 AS INTEGER) AS age_25_to_40,
 CAST(Age_in_years > 40 AS INTEGER) AS age_over_40,

  -- Checking account status
  CASE Status_of_existing_checking_account
    WHEN 'A11' THEN '<0'
    WHEN 'A12' THEN '0-200'
    WHEN 'A13' THEN '>=200_or_salary'
    WHEN 'A14' THEN 'no_account'
    ELSE 'unknown'
  END AS checking_desc,
  CAST(Status_of_existing_checking_account='A11' AS INTEGER) AS chk_lt_0,
  CAST(Status_of_existing_checking_account='A12' AS INTEGER) AS chk_0_200,
  CAST(Status_of_existing_checking_account='A13' AS INTEGER) AS chk_ge_200,
  CAST(Status_of_existing_checking_account='A14' AS INTEGER) AS chk_none,

  -- Credit history
  CASE Credit_history
    WHEN 'A30' THEN 'no_credits'
    WHEN 'A31' THEN 'all_paid_duly'
    WHEN 'A32' THEN 'existing_paid_duly'
    WHEN 'A33' THEN 'past_delay'
    WHEN 'A34' THEN 'critical_or_other'
    ELSE 'unknown'
  END AS history_desc,
  CAST(Credit_history='A30' AS INTEGER) AS hist_no_credits,
  CAST(Credit_history='A31' AS INTEGER) AS hist_all_paid,
  CAST(Credit_history='A32' AS INTEGER) AS hist_existing_paid,
  CAST(Credit_history='A33' AS INTEGER) AS hist_past_delay,
  CAST(Credit_history='A34' AS INTEGER) AS hist_critical,

  -- Purpose
  CASE Purpose
    WHEN 'A40' THEN 'car_new'
    WHEN 'A41' THEN 'car_used'
    WHEN 'A42' THEN 'furniture'
    WHEN 'A43' THEN 'radio_tv'
    WHEN 'A44' THEN 'domestic_app'
    WHEN 'A45' THEN 'repairs'
    WHEN 'A46' THEN 'education'
    WHEN 'A47' THEN 'vacation'
    WHEN 'A48' THEN 'retraining'
    WHEN 'A49' THEN 'business'
    WHEN 'A410' THEN 'others'
    ELSE 'unknown'
  END AS purpose_desc,
  CAST(Purpose='A40' AS INTEGER) AS pur_car_new,
  CAST(Purpose='A41' AS INTEGER) AS pur_car_used,
  CAST(Purpose='A42' AS INTEGER) AS pur_furniture,
  CAST(Purpose='A43' AS INTEGER) AS pur_radio_tv,
  CAST(Purpose='A44' AS INTEGER) AS pur_domestic,
  CAST(Purpose='A45' AS INTEGER) AS pur_repairs,
  CAST(Purpose='A46' AS INTEGER) AS pur_education,
  CAST(Purpose='A47' AS INTEGER) AS pur_vacation,
  CAST(Purpose='A48' AS INTEGER) AS pur_retraining,
  CAST(Purpose='A49' AS INTEGER) AS pur_business,
  CAST(Purpose='A410' AS INTEGER) AS pur_others,

  -- Savings
  CASE Savings_account_bonds
    WHEN 'A61' THEN '<100'
    WHEN 'A62' THEN '100-500'
    WHEN 'A63' THEN '500-1000'
    WHEN 'A64' THEN '>=1000'
    ELSE 'unknown'
  END AS savings_desc,
  CAST(Savings_account_bonds='A61' AS INTEGER) AS sav_lt_100,
  CAST(Savings_account_bonds='A62' AS INTEGER) AS sav_100_500,
  CAST(Savings_account_bonds='A63' AS INTEGER) AS sav_500_1000,
  CAST(Savings_account_bonds='A64' AS INTEGER) AS sav_ge_1000,
  CAST(Savings_account_bonds='A65' AS INTEGER) AS sav_unknown,

  -- Employment
  CASE Present_employment_since
    WHEN 'A71' THEN 'unemployed'
    WHEN 'A72' THEN '<1yr'
    WHEN 'A73' THEN '1-4yr'
    WHEN 'A74' THEN '4-7yr'
    WHEN 'A75' THEN '>=7yr'
    ELSE 'unknown'
  END AS emp_desc,
  CAST(Present_employment_since='A71' AS INTEGER) AS emp_unempl,
  CAST(Present_employment_since='A72' AS INTEGER) AS emp_lt1,
  CAST(Present_employment_since='A73' AS INTEGER) AS emp_1_4,
  CAST(Present_employment_since='A74' AS INTEGER) AS emp_4_7,
  CAST(Present_employment_since='A75' AS INTEGER) AS emp_ge7,

  -- Personal status & sex
  CASE Personal_status_and_sex
    WHEN 'A91' THEN 'm_div_sep'
    WHEN 'A92' THEN 'f_div_sep_mar'
    WHEN 'A93' THEN 'm_single'
    WHEN 'A94' THEN 'm_mar_wid'
    WHEN 'A95' THEN 'f_single'
    ELSE 'unknown'
  END AS status_sex_desc,
  CAST(Personal_status_and_sex='A91' AS INTEGER) AS sex_m_divsep,
  CAST(Personal_status_and_sex='A92' AS INTEGER) AS sex_f_divsep_mar,
  CAST(Personal_status_and_sex='A93' AS INTEGER) AS sex_m_single,
  CAST(Personal_status_and_sex='A94' AS INTEGER) AS sex_m_marwid,
  CAST(Personal_status_and_sex='A95' AS INTEGER) AS sex_f_single,

  -- Other debtors/guarantors
  CASE Other_debtors_guarantors
    WHEN 'A101' THEN 'none'
    WHEN 'A102' THEN 'coapplicant'
    WHEN 'A103' THEN 'guarantor'
    ELSE 'unknown'
  END AS debtors_desc,
  CAST(Other_debtors_guarantors='A101' AS INTEGER) AS debt_none,
  CAST(Other_debtors_guarantors='A102' AS INTEGER) AS debt_coapp,
  CAST(Other_debtors_guarantors='A103' AS INTEGER) AS debt_guarantor,

  -- Residence
  Present_residence_since,

  -- Property
  CASE Property
    WHEN 'A121' THEN 'real_estate'
    WHEN 'A122' THEN 'savings_ins'
    WHEN 'A123' THEN 'car_other'
    WHEN 'A124' THEN 'unknown'
    ELSE 'unknown'
  END AS property_desc,
  CAST(Property='A121' AS INTEGER) AS prop_real,
  CAST(Property='A122' AS INTEGER) AS prop_sav_ins,
  CAST(Property='A123' AS INTEGER) AS prop_car_other,
  CAST(Property='A124' AS INTEGER) AS prop_unknown,

  -- Other installment plans
  CASE Other_installment_plans
    WHEN 'A141' THEN 'bank'
    WHEN 'A142' THEN 'stores'
    WHEN 'A143' THEN 'none'
    ELSE 'unknown'
  END AS inst_plans_desc,
  CAST(Other_installment_plans='A141' AS INTEGER) AS inst_bank,
  CAST(Other_installment_plans='A142' AS INTEGER) AS inst_stores,
  CAST(Other_installment_plans='A143' AS INTEGER) AS inst_none,

  -- Housing
  CASE Housing
    WHEN 'A151' THEN 'rent'
    WHEN 'A152' THEN 'own'
    WHEN 'A153' THEN 'free'
    ELSE 'unknown'
  END AS housing_desc,
  CAST(Housing='A151' AS INTEGER) AS house_rent,
  CAST(Housing='A152' AS INTEGER) AS house_own,
  CAST(Housing='A153' AS INTEGER) AS house_free,

  -- Credits at bank
  Number_of_existing_credits_at_this_bank,

  -- Job
  CASE Job
    WHEN 'A171' THEN 'unskilled_nonres'
    WHEN 'A172' THEN 'unskilled_res'
    WHEN 'A173' THEN 'skilled'
    WHEN 'A174' THEN 'management'
    ELSE 'unknown'
  END AS job_desc,
  CAST(Job='A171' AS INTEGER) AS job_unsk_nonres,
  CAST(Job='A172' AS INTEGER) AS job_unsk_res,
  CAST(Job='A173' AS INTEGER) AS job_skilled,
  CAST(Job='A174' AS INTEGER) AS job_manage,

  -- Dependents
  Number_of_people_being_liable_to_provide_maintenance_for,

  -- Telephone
  CASE Telephone
    WHEN 'A191' THEN 'none'
    WHEN 'A192' THEN 'yes_registered'
    ELSE 'unknown'
  END AS tel_desc,
  CAST(Telephone='A191' AS INTEGER) AS tel_none,
  CAST(Telephone='A192' AS INTEGER) AS tel_yes,

  -- Foreign worker
  CASE Foreign_worker
    WHEN 'A201' THEN 'yes'
    WHEN 'A202' THEN 'no'
    ELSE 'unknown'
  END AS foreign_desc,
  CAST(Foreign_worker='A201' AS INTEGER) AS foreign_yes,
  CAST(Foreign_worker='A202' AS INTEGER) AS foreign_no

FROM german_credit_raw;
