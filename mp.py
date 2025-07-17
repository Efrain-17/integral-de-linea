import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import sympy as sp
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

def parsear_funcion(expr_str, variables=['x', 'y', 't']):
    """
    Convierte una cadena de texto en una función simbólica
    """
    try:
        # Definir variables simbólicas
        x, y, t = sp.symbols('x y t')
        
        # Reemplazar funciones comunes
        expr_str = expr_str.replace('^', '**')
        expr_str = expr_str.replace('sen', 'sin')
        expr_str = expr_str.replace('cos', 'cos')
        expr_str = expr_str.replace('ln', 'log')
        expr_str = expr_str.replace('e^', 'exp')
        
        # Parsear la expresión
        expr = sp.sympify(expr_str)
        return expr
    except:
        return None

def crear_funcion_numerica(expr, variables=['x', 'y', 't']):
    """
    Convierte una expresión simbólica en una función numérica
    """
    try:
        x, y, t = sp.symbols('x y t')
        return sp.lambdify((x, y, t), expr, 'numpy')
    except:
        return None

def calcular_integral_linea_escalar(f_expr, curva_x, curva_y, t_min, t_max):
    """
    Calcula la integral de línea de un campo escalar
    ∫_C f(x,y) ds
    """
    try:
        # Crear funciones numéricas
        f_num = crear_funcion_numerica(f_expr)
        x_num = crear_funcion_numerica(curva_x)
        y_num = crear_funcion_numerica(curva_y)
        
        # Calcular las derivadas de la curva
        t = sp.Symbol('t')
        dx_dt = sp.diff(curva_x, t)
        dy_dt = sp.diff(curva_y, t)
        
        dx_dt_num = crear_funcion_numerica(dx_dt)
        dy_dt_num = crear_funcion_numerica(dy_dt)
        
        def integrand(t_val):
            x_val = x_num(0, 0, t_val)
            y_val = y_num(0, 0, t_val)
            f_val = f_num(x_val, y_val, t_val)
            
            dx_val = dx_dt_num(0, 0, t_val)
            dy_val = dy_dt_num(0, 0, t_val)
            
            # ds = sqrt((dx/dt)² + (dy/dt)²) dt
            ds_dt = np.sqrt(dx_val**2 + dy_val**2)
            
            return f_val * ds_dt
        
        resultado, error = integrate.quad(integrand, t_min, t_max)
        return resultado, error
    except Exception as e:
        return None, str(e)

def calcular_integral_linea_vectorial(P_expr, Q_expr, curva_x, curva_y, t_min, t_max):
    """
    Calcula la integral de línea de un campo vectorial
    ∫_C P dx + Q dy
    """
    try:
        # Crear funciones numéricas
        P_num = crear_funcion_numerica(P_expr)
        Q_num = crear_funcion_numerica(Q_expr)
        x_num = crear_funcion_numerica(curva_x)
        y_num = crear_funcion_numerica(curva_y)
        
        # Calcular las derivadas de la curva
        t = sp.Symbol('t')
        dx_dt = sp.diff(curva_x, t)
        dy_dt = sp.diff(curva_y, t)
        
        dx_dt_num = crear_funcion_numerica(dx_dt)
        dy_dt_num = crear_funcion_numerica(dy_dt)
        
        def integrand(t_val):
            x_val = x_num(0, 0, t_val)
            y_val = y_num(0, 0, t_val)
            
            P_val = P_num(x_val, y_val, t_val)
            Q_val = Q_num(x_val, y_val, t_val)
            
            dx_val = dx_dt_num(0, 0, t_val)
            dy_val = dy_dt_num(0, 0, t_val)
            
            return P_val * dx_val + Q_val * dy_val
        
        resultado, error = integrate.quad(integrand, t_min, t_max)
        return resultado, error
    except Exception as e:
        return None, str(e)

def crear_grafico_curva(curva_x, curva_y, t_min, t_max, titulo="Curva"):
    """
    Crea el gráfico de la curva parametrizada
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generar puntos de la curva
        t_vals = np.linspace(t_min, t_max, 1000)
        x_num = crear_funcion_numerica(curva_x)
        y_num = crear_funcion_numerica(curva_y)
        
        x_vals = [x_num(0, 0, t) for t in t_vals]
        y_vals = [y_num(0, 0, t) for t in t_vals]
        
        # Dibujar la curva
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='Curva C')
        
        # Marcar puntos inicial y final
        ax.plot(x_vals[0], y_vals[0], 'go', markersize=8, label=f'Inicio t={t_min}')
        ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=8, label=f'Final t={t_max}')
        
        # Añadir flechas para mostrar dirección
        n_arrows = 5
        for i in range(n_arrows):
            idx = int(i * len(x_vals) / n_arrows)
            if idx < len(x_vals) - 1:
                dx = x_vals[idx + 1] - x_vals[idx]
                dy = y_vals[idx + 1] - y_vals[idx]
                ax.arrow(x_vals[idx], y_vals[idx], dx, dy, 
                        head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(titulo)
        ax.legend()
        ax.axis('equal')
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico: {e}")
        return None

def crear_grafico_campo_vectorial(P_expr, Q_expr, curva_x, curva_y, t_min, t_max):
    """
    Crea el gráfico del campo vectorial junto con la curva
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Generar puntos de la curva
        t_vals = np.linspace(t_min, t_max, 1000)
        x_num = crear_funcion_numerica(curva_x)
        y_num = crear_funcion_numerica(curva_y)
        
        x_vals = [x_num(0, 0, t) for t in t_vals]
        y_vals = [y_num(0, 0, t) for t in t_vals]
        
        # Dibujar la curva
        ax.plot(x_vals, y_vals, 'b-', linewidth=3, label='Curva C')
        
        # Crear malla para el campo vectorial
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        y_min, y_max = min(y_vals) - 1, max(y_vals) + 1
        
        x_mesh = np.linspace(x_min, x_max, 15)
        y_mesh = np.linspace(y_min, y_max, 15)
        X, Y = np.meshgrid(x_mesh, y_mesh)
        
        # Calcular el campo vectorial
        P_num = crear_funcion_numerica(P_expr)
        Q_num = crear_funcion_numerica(Q_expr)
        
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    U[i, j] = P_num(X[i, j], Y[i, j], 0)
                    V[i, j] = Q_num(X[i, j], Y[i, j], 0)
                except:
                    U[i, j] = 0
                    V[i, j] = 0
        
        # Dibujar el campo vectorial
        ax.quiver(X, Y, U, V, alpha=0.6, color='gray', scale_units='xy', scale=1)
        
        # Marcar puntos inicial y final
        ax.plot(x_vals[0], y_vals[0], 'go', markersize=10, label=f'Inicio t={t_min}')
        ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=10, label=f'Final t={t_max}')
        
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Campo Vectorial y Curva de Integración')
        ax.legend()
        
        return fig
    except Exception as e:
        st.error(f"Error al crear el gráfico del campo vectorial: {e}")
        return None

def main():
    st.title("🧮 Calculadora de Integral de Línea")
    st.write("Esta aplicación calcula integrales de línea para campos escalares y vectoriales.")
    
    # Seleccionar tipo de integral
    tipo_integral = st.selectbox(
        "Selecciona el tipo de integral:",
        ["Campo Escalar (∫_C f(x,y) ds)", "Campo Vectorial (∫_C P dx + Q dy)"]
    )
    
    st.subheader("📈 Definición de la Curva")
    st.write("Define la curva parametrizada C: r(t) = (x(t), y(t))")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_curva = st.text_input("x(t) =", value="cos(t)", help="Ejemplo: cos(t), t, t**2")
    
    with col2:
        y_curva = st.text_input("y(t) =", value="sin(t)", help="Ejemplo: sin(t), t, 2*t")
    
    col3, col4 = st.columns(2)
    
    with col3:
        t_min = st.number_input("t inicial:", value=0.0, step=0.1)
    
    with col4:
        t_max = st.number_input("t final:", value=2*np.pi, step=0.1)
    
    # Parsear la curva
    curva_x_expr = parsear_funcion(x_curva)
    curva_y_expr = parsear_funcion(y_curva)
    
    if curva_x_expr is None or curva_y_expr is None:
        st.error("❌ Error al parsear la curva. Verifica la sintaxis.")
        return
    
    # Campos según el tipo de integral
    if tipo_integral == "Campo Escalar (∫_C f(x,y) ds)":
        st.subheader("📊 Campo Escalar")
        f_input = st.text_input("f(x,y) =", value="x**2 + y**2", 
                               help="Ejemplo: x**2 + y**2, x*y, sin(x) + cos(y)")
        
        f_expr = parsear_funcion(f_input)
        if f_expr is None:
            st.error("❌ Error al parsear la función. Verifica la sintaxis.")
            return
        
        # Calcular la integral
        if st.button("🔢 Calcular Integral"):
            resultado, error = calcular_integral_linea_escalar(f_expr, curva_x_expr, curva_y_expr, t_min, t_max)
            
            if resultado is not None:
                st.success(f"✅ Resultado: **{resultado:.6f}**")
                st.info(f"Error estimado: {error:.2e}")
                
                # Mostrar la fórmula
                st.latex(r"\int_C f(x,y) \, ds = \int_{" + str(t_min) + r"}^{" + str(t_max) + r"} f(x(t), y(t)) \sqrt{\left(\frac{dx}{dt}\right)^2 + \left(\frac{dy}{dt}\right)^2} \, dt")
            else:
                st.error(f"❌ Error en el cálculo: {error}")
    
    else:  # Campo Vectorial
        st.subheader("🧭 Campo Vectorial")
        col5, col6 = st.columns(2)
        
        with col5:
            P_input = st.text_input("P(x,y) =", value="x", help="Componente x del campo vectorial")
        
        with col6:
            Q_input = st.text_input("Q(x,y) =", value="y", help="Componente y del campo vectorial")
        
        P_expr = parsear_funcion(P_input)
        Q_expr = parsear_funcion(Q_input)
        
        if P_expr is None or Q_expr is None:
            st.error("❌ Error al parsear el campo vectorial. Verifica la sintaxis.")
            return
        
        # Calcular la integral
        if st.button("🔢 Calcular Integral"):
            resultado, error = calcular_integral_linea_vectorial(P_expr, Q_expr, curva_x_expr, curva_y_expr, t_min, t_max)
            
            if resultado is not None:
                st.success(f"✅ Resultado: **{resultado:.6f}**")
                st.info(f"Error estimado: {error:.2e}")
                
                # Mostrar la fórmula
                st.latex(r"\int_C P \, dx + Q \, dy = \int_{" + str(t_min) + r"}^{" + str(t_max) + r"} \left[P(x(t), y(t))\frac{dx}{dt} + Q(x(t), y(t))\frac{dy}{dt}\right] dt")
            else:
                st.error(f"❌ Error en el cálculo: {error}")
    
    # Visualización
    st.subheader("📊 Visualización")
    mostrar_grafico = st.checkbox("Mostrar gráfico", value=True)
    
    if mostrar_grafico:
        if tipo_integral == "Campo Escalar (∫_C f(x,y) ds)":
            fig = crear_grafico_curva(curva_x_expr, curva_y_expr, t_min, t_max, "Curva de Integración")
        else:
            fig = crear_grafico_campo_vectorial(P_expr, Q_expr, curva_x_expr, curva_y_expr, t_min, t_max)
        
        if fig is not None:
            st.pyplot(fig)
    
    # Información adicional
    with st.expander("ℹ️ Información sobre Integrales de Línea"):
        st.write("""
        **Integral de Línea de un Campo Escalar:**
        
        ∫_C f(x,y) ds = ∫_a^b f(x(t), y(t)) √[(dx/dt)² + (dy/dt)²] dt
        
        Representa la suma de los valores de f a lo largo de la curva C, ponderada por el elemento de longitud ds.
        
        **Integral de Línea de un Campo Vectorial:**
        
        ∫_C P dx + Q dy = ∫_a^b [P(x(t), y(t))(dx/dt) + Q(x(t), y(t))(dy/dt)] dt
        
        Representa el trabajo realizado por el campo vectorial F = (P, Q) a lo largo de la curva C.
        
        **Sintaxis de funciones:**
        - Potencias: x**2, y**3
        - Trigonométricas: sin(x), cos(y), tan(x)
        - Exponencial: exp(x) o e**x
        - Logaritmo: log(x)
        - Constantes: pi, e
        """)
    
    # Ejemplos
    with st.expander("📝 Ejemplos"):
        st.write("""
        **Ejemplo 1: Círculo unitario**
        - Curva: x(t) = cos(t), y(t) = sin(t)
        - Intervalo: [0, 2π]
        - Campo escalar: f(x,y) = x² + y²
        
        **Ejemplo 2: Segmento de recta**
        - Curva: x(t) = t, y(t) = 2t
        - Intervalo: [0, 1]
        - Campo vectorial: P(x,y) = x, Q(x,y) = y
        
        **Ejemplo 3: Parábola**
        - Curva: x(t) = t, y(t) = t²
        - Intervalo: [0, 2]
        - Campo escalar: f(x,y) = x*y
        """)

if __name__ == "__main__":
    main()