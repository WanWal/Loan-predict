import pickle
import pandas as pd

class LoanApprovalModel:
    def __init__(self, model_path='model.pkl'):
        # Memuat model dan encoder yang telah disimpan sebagai dictionary
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']   # Model yang dilatih
                self.encoders = data['encoders']  # Encoder untuk data kategorikal
        except FileNotFoundError:
            raise FileNotFoundError(f"File model tidak ditemukan di path: {model_path}")
        except Exception as e:
            raise ValueError(f"Terjadi kesalahan saat memuat model: {e}")

    def predict(self, input_data: dict):
        # Mengubah input_data menjadi DataFrame
        df = pd.DataFrame([input_data])

        # Lakukan encoding untuk kolom-kolom kategorikal
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col])
                except Exception as e:
                    raise ValueError(f"Encoding gagal di kolom '{col}': {e}")
            else:
                raise ValueError(f"Kolom '{col}' tidak ditemukan di data input.")

        # Melakukan prediksi
        try:
            prediction = self.model.predict(df)[0]
            return prediction
        except Exception as e:
            raise ValueError(f"Prediksi gagal: {e}")
