import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title='AI Job Market Insights')
st.title('Analisis Pasar Kerja Berbasis AI 📊')
st.subheader('Upload Dataset')

# Upload file Excel
uploaded_file = st.file_uploader('Pilih file XLSX', type='xlsx')
if uploaded_file:
    st.markdown('---')
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    st.dataframe(df)

    if st.button("Info Dataset"):
        with st.expander("Informasi Dataset", expanded=True):
            # Menampilkan informasi dataset dalam bentuk float page
            st.write("### Info Umum Dataset")
            st.write(f"Jumlah Baris: {df.shape[0]}")
            st.write(f"Jumlah Kolom: {df.shape[1]}")
            st.write("### Tipe Data")
            st.write(df.dtypes)
            
            # Tombol untuk menutup informasi dataset
            if st.button("Tutup Info"):
                st.empty()  # Menutup expander atau halaman info

    # Preprocessing
    st.subheader("Preprocessing Data")
    if 'preprocessed_data' not in st.session_state:
        st.session_state['preprocessed_data'] = None

    if st.button("Lakukan Preprocessing"):
        df = df.dropna()  # Menghapus nilai kosong
        scaler = StandardScaler()
        columns_to_scale = ['Salary_USD']  # Kolom yang diskalakan
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        st.session_state['preprocessed_data'] = df
        st.write("Data setelah preprocessing:")
        st.dataframe(df)

    # Clustering
    st.subheader("Analisis Data - Clustering")
    if st.session_state['preprocessed_data'] is not None:
        k = st.slider("Pilih Jumlah Cluster", 2, 10, 3)
        df = st.session_state['preprocessed_data']
        
        if st.button("Lakukan Clustering (K-Means)"):
            model = KMeans(n_clusters=k)
            df['Cluster'] = model.fit_predict(df[['Salary_USD']])
            st.session_state['clustered_data'] = df

            st.write(f"Data dengan Cluster K-Means (k={k}):")
            st.dataframe(df)
            st.write("Centroid Cluster:")
            st.write(pd.DataFrame(model.cluster_centers_, columns=['Salary_USD']))

    # Visualisasi
    if 'clustered_data' in st.session_state:
        st.subheader("Visualisasi Data")
        df = st.session_state['clustered_data']
        visual_options = [
            "Distribusi Gaji berdasarkan Jabatan Pekerjaan",
            "Perbandingan Gaji antar Industri",
            "Hubungan antara Risiko Otomatisasi dan Gaji",
            "Hubungan antara Ukuran Perusahaan dan Risiko Otomatisasi",
            "Distribusi Lokasi Kerja Berdasarkan Remote Friendly dan Industri",
            "Persentase Pekerjaan yang Mendukung Kerja Jarak Jauh",
            "Distribusi Gaji berdasarkan Ukuran Perusahaan",
            "Hubungan Proyeksi Pertumbuhan dengan Tingkat Adopsi AI",
            "Keterampilan yang Paling Banyak Dibutuhkan",
            "Distribusi Gaji Berdasarkan Lokasi",
            "Rata-rata Gaji untuk Setiap Tingkat Adopsi AI",
            "Perbandingan Risiko Otomatisasi di Berbagai Lokasi",
            "Proyeksi Pertumbuhan Pekerjaan Berdasarkan Industri",
            "Frekuensi Lokasi Kerja berdasarkan Remote Friendly",
            "Distribusi Pekerjaan Berdasarkan Ukuran Perusahaan",
            "Hubungan Gaji dengan Jumlah Keterampilan yang Dibutuhkan",
            "Distribusi Jabatan Berdasarkan Industri",
            "Korelasi antara Gaji dan Risiko Otomatisasi",
            "Frekuensi Jabatan yang Mendukung Kerja Remote",
            "Distribusi Gaji berdasarkan Proyeksi Pertumbuhan Pekerjaan",
            "Frekuensi Jabatan Berdasarkan Industri",
        ]
        viz_choice = st.selectbox("Pilih Visualisasi", visual_options)

        # Distribusi Gaji berdasarkan Jabatan Pekerjaan
        if viz_choice == visual_options[0]:
            fig = px.histogram(
                df, 
                x='Salary_USD', 
                color='Job_Title', 
                title="Distribusi Gaji berdasarkan Jabatan Pekerjaan",
                nbins=30,  # Jumlah bin histogram (opsional)
                barmode='overlay'  # Tumpang tindih histogram untuk semua jabatan
            )
            st.plotly_chart(fig)

        # Perbandingan Gaji antar Industri
        elif viz_choice == visual_options[1]:
            fig = px.bar(df, x='Industry', y='Salary_USD', color='Cluster', title="Perbandingan Gaji antar Industri")
            st.plotly_chart(fig)

        # Hubungan antara Risiko Otomatisasi dan Gaji
        elif viz_choice == visual_options[2]:
            fig = px.scatter(df, x='Automation_Risk', y='Salary_USD', color='Cluster', title="Hubungan Risiko Otomatisasi dan Gaji")
            st.plotly_chart(fig)

        # Tingkat Adopsi AI dalam Berbagai Industri
        elif viz_choice == visual_options[3]:
            df['Salary_Size'] = df['Salary_USD'] + abs(df['Salary_USD'].min()) + 1  # Geser semua nilai ke non-negatif
            fig = px.scatter(
                df, x='Company_Size', y='Automation_Risk', size='Salary_Size', color='Industry',
                title="Hubungan antara Ukuran Perusahaan dan Risiko Otomatisasi",
                hover_data=['Job_Title']
            )
            st.plotly_chart(fig)

        # Distribusi Lokasi berdasarkan Industri
        elif viz_choice == visual_options[4]:
            fig = px.bar(
                df, x='Location', color='Industry', barmode='stack',
                facet_col='Remote_Friendly', title="Distribusi Lokasi Kerja Berdasarkan Remote Friendly dan Industri"
            )
            st.plotly_chart(fig)

        # Persentase Pekerjaan yang Mendukung Kerja Jarak Jauh
        elif viz_choice == visual_options[5]:
            fig = px.pie(df, names='Remote_Friendly', title="Persentase Pekerjaan yang Mendukung Kerja Jarak Jauh")
            st.plotly_chart(fig)

        # Distribusi Gaji berdasarkan Ukuran Perusahaan
        elif viz_choice == visual_options[6]:
            fig = px.box(df, x='Company_Size', y='Salary_USD', title="Distribusi Gaji berdasarkan Ukuran Perusahaan")
            st.plotly_chart(fig)

        # Hubungan Proyeksi Pertumbuhan dengan Tingkat Adopsi AI
        elif viz_choice == visual_options[7]:
            fig = px.bar(df, x='AI_Adoption_Level', y='Job_Growth_Projection', color='Industry', title="Hubungan Proyeksi Pertumbuhan dengan Tingkat Adopsi AI")
            st.plotly_chart(fig)

        # Keterampilan yang Paling Banyak Dibutuhkan
        elif viz_choice == visual_options[8]:
            skill_words = " ".join(df['Required_Skills'].dropna())
            wordcloud = WordCloud(width=800, height=400).generate(skill_words)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

        # Distribusi Gaji Berdasarkan Jabatan dan Lokasi
        elif viz_choice == visual_options[9]:  # Indeks visualisasi tetap
            fig = px.histogram(
                df, 
                x='Location', 
                y='Salary_USD', 
                title="Distribusi Gaji Berdasarkan Lokasi",
                nbins=30,  # Jumlah bin histogram (opsional)
                histfunc='avg'  # Fungsi agregasi, rata-rata gaji per lokasi
            )
            st.plotly_chart(fig)

        # Rata-rata Gaji untuk Setiap Tingkat Adopsi AI
        elif viz_choice == visual_options[10]:
            avg_salary = df.groupby('AI_Adoption_Level')['Salary_USD'].mean().reset_index()
            fig = px.bar(avg_salary, x='AI_Adoption_Level', y='Salary_USD', title="Rata-rata Gaji untuk Setiap Tingkat Adopsi AI")
            st.plotly_chart(fig)

        # Perbandingan Risiko Otomatisasi di Berbagai Lokasi (Grouped Bar Chart)
        elif viz_choice == visual_options[11]:
            fig = px.bar(df, x='Location', y='Automation_Risk', color='Industry', barmode='group',
                         title="Perbandingan Risiko Otomatisasi di Berbagai Lokasi")
            st.plotly_chart(fig)

        # Proyeksi Pertumbuhan Pekerjaan Berdasarkan Industri (Clustered Column Chart)
        elif viz_choice == visual_options[12]:
            fig = px.bar(df, x='Industry', y='Job_Growth_Projection', color='Industry',
                         title="Proyeksi Pertumbuhan Pekerjaan Berdasarkan Industri")
            st.plotly_chart(fig)

        # Frekuensi Lokasi Kerja berdasarkan Remote Friendly (Bar Chart)
        elif viz_choice == visual_options[13]:
            remote_counts = df['Remote_Friendly'].value_counts().reset_index()
            remote_counts.columns = ['Remote_Friendly', 'Count']
            
            # Membuat bar chart dengan warna berbeda berdasarkan kategori 'Remote_Friendly'
            fig = px.bar(
                remote_counts, 
                x='Remote_Friendly', 
                y='Count', 
                color='Remote_Friendly',  # Warna berdasarkan kategori Remote_Friendly
                title="Frekuensi Lokasi Kerja berdasarkan Remote Friendly",
                text='Count',  # Menambahkan label dengan jumlah frekuensi
                color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"]  # Mengatur warna khusus (opsional)
            )

            # Menambahkan label pada sumbu dan menyesuaikan ukuran font
            fig.update_layout(
                xaxis_title="Tipe Lokasi Kerja", 
                yaxis_title="Jumlah Frekuensi", 
                font=dict(size=14),  # Ukuran font untuk keseluruhan
                title_font=dict(size=18),  # Ukuran font untuk judul
                legend_title="Remote Friendly",  # Judul untuk legenda
            )

            # Menambahkan posisi teks di luar batang
            fig.update_traces(
                texttemplate='%{text}',  # Menampilkan angka pada setiap batang
                textposition='outside',  # Posisi teks di luar batang
                marker=dict(line=dict(width=2, color='DarkSlateGrey'))  # Menambahkan outline pada batang
            )

            st.plotly_chart(fig)

        # Distribusi Pekerjaan Berdasarkan Ukuran Perusahaan (Pie Chart)
        elif viz_choice == visual_options[14]:
            fig = px.pie(df, names='Company_Size', title="Distribusi Pekerjaan Berdasarkan Ukuran Perusahaan")
            st.plotly_chart(fig)

                # Hubungan Gaji dengan Jumlah Keterampilan yang Dibutuhkan
        elif viz_choice == visual_options[15]:
            df['Skill_Count'] = df['Required_Skills'].str.split().apply(len)
            fig = px.scatter(
                df, x='Skill_Count', y='Salary_USD', color='Industry',
                title="Hubungan Gaji dengan Jumlah Keterampilan yang Dibutuhkan",
                hover_data=['Job_Title']
            )
            st.plotly_chart(fig)

        # Distribusi Jabatan Berdasarkan Industri (Stacked Bar Chart)
        elif viz_choice == visual_options[16]:
            fig = px.bar(df, x='Industry', y='Job_Title', color='Industry', text='Job_Title', barmode='stack',
                         title="Distribusi Jabatan Berdasarkan Industri")
            st.plotly_chart(fig)

        # Korelasi antara Gaji dan Risiko Otomatisasi (Heatmap)
        elif viz_choice == visual_options[17]:
            # Konversi kolom Automation_Risk ke nilai numerik
            df['Automation_Risk_Numeric'] = df['Automation_Risk'].map({'Low': 1, 'Medium': 2, 'High': 3})
            
            # Hitung korelasi
            correlation_data = df[['Salary_USD', 'Automation_Risk_Numeric']].corr()
            
            # Visualisasi Heatmap
            fig = px.imshow(correlation_data, text_auto=True, title="Korelasi antara Gaji dan Risiko Otomatisasi")
            st.plotly_chart(fig)

        # Frekuensi Jabatan yang Mendukung Kerja Remote (Bar Chart)
        elif viz_choice == visual_options[18]:
            remote_jobs = df[df['Remote_Friendly'] == 'Yes']['Job_Title'].value_counts().reset_index()
            remote_jobs.columns = ['Job_Title', 'Count']
            fig = px.bar(remote_jobs, x='Job_Title', y='Count', title="Frekuensi Jabatan yang Mendukung Kerja Remote")
            st.plotly_chart(fig)

        # Distribusi Gaji berdasarkan Proyeksi Pertumbuhan Pekerjaan (Violin Plot)
        elif viz_choice == visual_options[19]:
            fig = px.violin(df, x='Job_Growth_Projection', y='Salary_USD', color='Cluster',
                            title="Distribusi Gaji berdasarkan Proyeksi Pertumbuhan Pekerjaan")
            st.plotly_chart(fig)


        # Frekuensi Jabatan Berdasarkan Industri
        elif viz_choice == visual_options[20]:
            pivot_data = df.pivot_table(
                index='Industry',
                columns='Job_Title',
                values='Salary_USD',
                aggfunc='count',  # Menghitung jumlah kemunculan
                fill_value=0  # Mengisi nilai kosong dengan 0
            )
            
            # Visualisasi Heatmap
            fig = px.imshow(
                pivot_data,
                color_continuous_scale='Viridis',  # Skema warna heatmap
                labels=dict(x="Job Title", y="Industry", color="Frequency"),
                title="Frekuensi Jabatan Berdasarkan Industri"
            )
            st.plotly_chart(fig)

