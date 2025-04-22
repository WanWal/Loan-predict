import streamlit as st
from oop import LoanApprovalModel

# Membuat model LoanApprovalModel
model = LoanApprovalModel('model.pkl')

# Judul aplikasi
st.title('Prediksi Persetujuan Pinjaman')

# Form input data
with st.form(key='loan_form'):
    age = st.number_input('Umur', min_value=18, max_value=100)
    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    education = st.selectbox('Pendidikan', ['High School', 'College', 'Graduate', 'Other'])
    income = st.number_input('Pendapatan', min_value=0)
    emp_exp = st.number_input('Pengalaman Kerja (tahun)', min_value=0)
    home = st.selectbox('Status Tempat Tinggal', ['RENT', 'OWN', 'MORTGAGE', 'OTHER'])
    loan_amnt = st.number_input('Jumlah Pinjaman', min_value=0)
    intent = st.selectbox('Tujuan Pinjaman', ['debt_consolidation', 'home_improvement', 'credit_card', 'other'])
    interest = st.number_input('Bunga Pinjaman (%)', min_value=0.0)
    percent_income = st.number_input('Persen Pendapatan untuk Pinjaman', min_value=0.0)
    cb_length = st.number_input('Lama Riwayat Kredit', min_value=0)
    score = st.number_input('Skor Kredit', min_value=300)
    default = st.selectbox('Pernah Gagal Bayar?', ['Yes', 'No'])

    submit = st.form_submit_button('Prediksi')

    if submit:
        # Data yang dikirim ke model
        data = {
            'person_age': age,
            'person_gender': gender,
            'person_education': education,
            'person_income': income,
            'person_emp_exp': emp_exp,
            'person_home_ownership': home,
            'loan_amnt': loan_amnt,
            'loan_intent': intent,
            'loan_int_rate': interest,
            'loan_percent_income': percent_income,
            'cb_person_cred_hist_length': cb_length,
            'credit_score': score,
            'previous_loan_defaults_on_file': default
        }

        # Prediksi menggunakan model
        try:
            result = model.predict(data)
            # Menampilkan hasil prediksi
            st.success('Status Pinjaman: **Disetujui**' if result == 1 else 'Status Pinjaman: **Ditolak**')
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
