import os
import numpy as np
import pandas as pd

def generate_credit_data(num_samples=2500, seed=42):
    """
    Generates a realistic synthetic credit card application dataset.
    """
    np.random.seed(seed)
    
    # 1. Generate demographic details
    genders = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.48, 0.52])
    ages = np.random.randint(18, 76, size=num_samples)
    
    marital_statuses = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'],
        size=num_samples,
        p=[0.35, 0.50, 0.12, 0.03]
    )
    
    education_levels = np.random.choice(
        ['High School', 'Graduate', 'Postgraduate', 'Doctorate'],
        size=num_samples,
        p=[0.30, 0.50, 0.15, 0.05]
    )
    
    # 2. Generate employment details
    employment_types = np.random.choice(
        ['Salaried', 'Self-Employed', 'Retired', 'Unemployed'],
        size=num_samples,
        p=[0.65, 0.20, 0.10, 0.05]
    )
    
    # Refine Age and Employment details for realism
    employment_durations = []
    annual_incomes = []
    
    for i in range(num_samples):
        emp_type = employment_types[i]
        age = ages[i]
        
        # Ensure Retired are mostly older, Unemployed have 0 duration, etc.
        if emp_type == 'Retired':
            if age < 55:
                ages[i] = np.random.randint(55, 76)
                age = ages[i]
            emp_dur = np.random.uniform(15, age - 18)
            income = np.random.uniform(15000, 80000)
        elif emp_type == 'Unemployed':
            emp_dur = 0.0
            income = np.random.uniform(5000, 20000)
            # Make sure age is logical
        elif emp_type == 'Self-Employed':
            max_dur = max(0, age - 20)
            emp_dur = np.random.uniform(0.5, max_dur) if max_dur > 0.5 else 0.5
            income = np.random.uniform(25000, 300000)
        else: # Salaried
            max_dur = max(0, age - 18)
            emp_dur = np.random.uniform(0.5, max_dur) if max_dur > 0.5 else 0.5
            income = np.random.uniform(20000, 250000)
            
        # Boost income for higher education
        edu = education_levels[i]
        if edu == 'Graduate':
            income *= 1.2
        elif edu == 'Postgraduate':
            income *= 1.5
        elif edu == 'Doctorate':
            income *= 1.8
            
        # Clip income to reasonable bounds
        income = np.clip(income, 5000, 450000)
        
        employment_durations.append(round(emp_dur, 1))
        annual_incomes.append(round(income, 2))
        
    annual_incomes = np.array(annual_incomes)
    employment_durations = np.array(employment_durations)
    
    # 3. Generate financial attributes
    # Loan amount requested
    loan_amounts = []
    for inc in annual_incomes:
        # Loan requested is typically proportion of income
        ratio = np.random.beta(2, 5) * 0.8  # ranges mostly from 0.05 to 0.5
        loan_amounts.append(round(inc * ratio, 2))
    loan_amounts = np.array(loan_amounts)
    
    # Existing debt
    existing_debts = []
    for inc in annual_incomes:
        # Debt ratio
        ratio = np.random.exponential(scale=0.15)
        existing_debts.append(round(inc * ratio, 2))
    existing_debts = np.array(existing_debts)
    
    # Credit Score (FICO range 300 - 850)
    # Higher income and older age correlates with better credit score
    base_scores = 500 + (ages / 75) * 100 + (annual_incomes / 450000) * 100
    noise_scores = np.random.normal(0, 70, size=num_samples)
    credit_scores = np.clip(base_scores + noise_scores, 300, 850).astype(int)
    
    # Credit History (1 = Good, 0 = Bad)
    # Strongly correlated with credit score
    credit_histories = []
    for cs in credit_scores:
        if cs >= 700:
            p_good = 0.95
        elif cs >= 600:
            p_good = 0.75
        elif cs >= 500:
            p_good = 0.40
        else:
            p_good = 0.10
        credit_histories.append(np.random.choice([1, 0], p=[p_good, 1 - p_good]))
    credit_histories = np.array(credit_histories)
    
    # Payment History Status
    payment_histories = []
    for ch in credit_histories:
        if ch == 1:
            p_ontime = 0.90
        else:
            p_ontime = 0.25
        payment_histories.append(np.random.choice(['On-time', 'Delayed'], p=[p_ontime, 1 - p_ontime]))
    payment_histories = np.array(payment_histories)
    
    # Credit Inquiries in last 6 months
    # People with lower credit score or higher debts tend to inquire more
    inquiries_base = np.random.poisson(lam=1.5, size=num_samples)
    for i in range(num_samples):
        if credit_scores[i] < 600:
            inquiries_base[i] += np.random.randint(1, 4)
    credit_inquiries = np.clip(inquiries_base, 0, 10)
    
    # 4. Generate Target (Approved = 1, Rejected = 0)
    approved = []
    for i in range(num_samples):
        score = 0.0
        
        # Credit Score impact
        cs = credit_scores[i]
        if cs >= 750:
            score += 5.0
        elif cs >= 680:
            score += 2.5
        elif cs >= 600:
            score += 0.5
        elif cs >= 500:
            score -= 2.0
        else:
            score -= 5.5
            
        # Credit History impact
        ch = credit_histories[i]
        if ch == 1:
            score += 3.5
        else:
            score -= 5.0
            
        # Debt-to-Income (DTI) ratio
        dti = existing_debts[i] / max(1000, annual_incomes[i])
        if dti > 0.45:
            score -= 4.0
        elif dti > 0.30:
            score -= 1.5
        elif dti < 0.15:
            score += 1.5
            
        # Loan-to-Income (LTI) ratio
        lti = loan_amounts[i] / max(1000, annual_incomes[i])
        if lti > 0.50:
            score -= 2.5
        elif lti <= 0.20:
            score += 1.0
            
        # Employment Type impact
        emp = employment_types[i]
        dur = employment_durations[i]
        if emp == 'Unemployed':
            score -= 4.5
        elif emp == 'Salaried':
            if dur > 5:
                score += 1.5
            elif dur > 2:
                score += 0.8
        elif emp == 'Self-Employed':
            if dur > 5:
                score += 1.0
            else:
                score -= 0.5
        elif emp == 'Retired':
            score += 0.5
            
        # Inquiries impact
        inq = credit_inquiries[i]
        if inq >= 5:
            score -= 3.5
        elif inq >= 3:
            score -= 1.5
        elif inq <= 1:
            score += 1.0
            
        # Payment History status
        ph = payment_histories[i]
        if ph == 'Delayed':
            score -= 2.5
        else:
            score += 1.0
            
        # Add random noise
        score += np.random.normal(0, 1.2)
        
        # Binary assignment (Threshold = 0.5)
        approved.append(1 if score > 0.5 else 0)
        
    approved = np.array(approved)
    
    # Print approval rate
    approval_rate = np.mean(approved) * 100
    print(f"Dataset generated. Size: {num_samples} records. Approval Rate: {approval_rate:.2f}%")
    
    # 5. Create DataFrame
    df = pd.DataFrame({
        'Applicant_ID': [f"APP-{10000 + i}" for i in range(num_samples)],
        'Gender': genders,
        'Age': ages,
        'Marital_Status': marital_statuses,
        'Education_Level': education_levels,
        'Employment_Type': employment_types,
        'Employment_Duration_Years': employment_durations,
        'Annual_Income': annual_incomes,
        'Loan_Amount': loan_amounts,
        'Existing_Debt': existing_debts,
        'Credit_Score': credit_scores,
        'Credit_History': credit_histories,
        'Payment_History_Status': payment_histories,
        'Credit_Inquiries': credit_inquiries,
        'Approved': approved
    })
    
    return df

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    df = generate_credit_data()
    output_path = os.path.join('data', 'credit_card_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved successfully to {output_path}")
