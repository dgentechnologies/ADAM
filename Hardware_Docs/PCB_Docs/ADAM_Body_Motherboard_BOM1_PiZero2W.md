# ADAM Body Motherboard — Production Bill of Materials (BOM 1)
## Option 1: Raspberry Pi Zero 2 W + All-Chip Discrete Peripheral ICs

**Company:** Dgen Technologies Pvt. Ltd. (Kolkata, India)  
**Product:** ADAM — Autonomous Desktop AI Module  
**Board:** Body Shell Motherboard (BOM 1 — Baseline / Cost-Optimized)  
**Compute Core:** Raspberry Pi Zero 2 W (Mounted via low-profile SMD/Berg headers)  
**PCB Layers:** 2-Layer or 4-Layer FR4 (1.6mm thickness, 1oz copper)  
**Revision:** V1.2 (Hardware & Reliability Hardened)  
**Date:** 13 September 2026  
**Local Sourcing Hub:** Chandni Chowk / Princep St., Kolkata & Authorized Indian Distributors (Robu.in, Quartz Components, Evelta, Mouser India)

---

## 1. System Overview & Architecture

BOM 1 integrates the complete Raspberry Pi Zero 2 W board onto a custom carrier motherboard containing **100% discrete chip implementations** for all supporting subsystems:
- **Dual-Stage Power & Servo Inrush Protection:** Direct 5V USB-C power entry with 3.0A primary polyfuse, AO3401A P-MOSFET reverse protection, **2200µF 16V input reservoir capacitor**, plus a **dedicated local 1000µF 16V low-ESR capacitor, SS34 flyback Schottky diode, and 1.5A isolation polyfuse directly at the pan servo header (`J_SERVO`)** to eliminate motor-induced brownout reboots.
- **Synchronous Audio ADC Subsystem:** Dual analog condenser microphone inputs with discrete 3.3V bias, **TI LMV358** dual op-amp preamplifier, and **TI PCM1808** 24-bit 96kHz stereo I2S ADC. Configured in **I2S Slave Mode** with explicit strapping resistors (`FMT=GND`, `MD0=GND`, `MD1=GND`, `BYPAS=GND`) and synchronous master clock driven by Raspberry Pi **`GPCLK0` (GPIO4)** to eliminate clock drift, clicks, and dropouts.
- **Onboard Audio Output:** Discrete **Maxim MAX98357AETE+T** 3.2W Class-D I2S amplifier IC with **`GAIN_SLOT` explicitly tied to `GND` via a 100kΩ resistor (+12dB fixed gain)** to prevent floating-pin volume drift.
- **Shielded Neck Link:** 4-pin locking JST-PH connector utilizing high-flex twisted shielded cabling (28 AWG) for reliable 921,600 baud communication with the ESP32-S3 Head Motherboard.

---

## 2. Complete Bill of Materials (BOM Table)

| Item # | Designator | Component / Description | Manufacturer Part Number (MPN) | Package / Footprint | Qty | Unit Price (INR ₹) | Primary Source (India / Kolkata) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **A. COMPUTE & MOUNTING** | | | | | | | |
| 1 | U1 | Raspberry Pi Zero 2 W (Quad Cortex-A53 @ 1.0 GHz, 512MB RAM, Wi-Fi/BT) | SC0510 / RPI-ZERO-2W | Module (65 × 30 mm) | 1 | ₹2,150.00 | Robu.in / Supreme Electronics (Chandni Chowk) |
| 2 | J1A, J1B | 2×20 Pin Dual-Row Female SMD Header (2.54mm pitch, low profile 5.0mm height) | 2X20-FEMALE-SMD-2.54 | SMD Dual Row 2×20 | 1 | ₹35.00 | Quartz Components / Robu.in |
| **B. POWER INPUT & PROTECTION** | | | | | | | |
| 3 | J_PWR | USB Type-C Receptacle 16-Pin / 6-Pin (Power Only, 5V @ 3A–5A rated) | TYPE-C-16P-SMD | SMD Hybrid 16-Pin | 1 | ₹18.00 | Evelta / Robu / Chandni Chowk |
| 4 | R1, R2 | 5.1kΩ 1% 1/10W Resistors (USB Type-C CC1 / CC2 Pulldowns) | RC0603FR-075K1L | 0603 SMD | 2 | ₹1.50 | Local passives trader / Robu |
| 5 | F1 | SMD Resettable PTC Polyfuse (3.0A Hold, 3.5A–6.0A Trip, 6V min) - Primary Power | SMD1812P300TF / 1812L300 | 1812 SMD | 1 | ₹16.00 | Evelta / Mouser India |
| 6 | Q1 | P-Channel MOSFET (30V, 4.2A, Rds(on) < 50mΩ) Reverse Polarity Protection | AO3401A / DMG3415U | SOT-23-3 | 1 | ₹6.00 | Quartz / Evelta / Chandni Chowk |
| 7 | D1 | TVS Diode Array (ESD Protection for USB & VBUS, 5V unidirectional) | USBLC6-2SC6 / SMBJ5.0A | SOT-23-6 / SMB | 1 | ₹12.00 | Evelta / Mouser India |
| 8 | C_BULK_IN | **2200µF 16V Low-ESR Radial Electrolytic Capacitor** (Input Power Reservoir) | UPW1C222MHD / EEU-FR1C222 | Radial (10×20mm, 5mm pitch) | 1 | ₹24.00 | Supreme Electronics / Classic Electronics (Chandni) |
| 9 | C1, C2, C3 | 10µF 16V X5R Ceramic Decoupling Capacitors | CL10A106KP8NNNC | 0603 SMD | 3 | ₹3.50 | Local passives / Robu |
| 10 | C4, C5, C6 | 100nF (0.1µF) 50V X7R Ceramic Decoupling Capacitors | CC0603KRX7R9BB104 | 0603 SMD | 3 | ₹1.50 | Local passives / Robu |
| **C. SERVO POINT-OF-LOAD INRUSH & BACK-EMF PROTECTION** | | | | | | | |
| 11 | F_SERVO | SMD Resettable PTC Polyfuse (0.75A Hold, 1.5A Trip) - Servo Subsystem Isolation | SMD1206P075TF | 1206 SMD | 1 | ₹9.00 | Evelta / Mouser India |
| 12 | C_SERVO_LOC| **1000µF 16V Low-ESR Radial Electrolytic Capacitor** (Mounted at J_SERVO) | UPW1C102MPD / EEU-FR1C102 | Radial (8×15mm, 3.5mm pitch) | 1 | ₹14.00 | Supreme Electronics / Chandni market |
| 13 | D_SERVO | 3A 40V Schottky Barrier Diode (Flyback / Back-EMF Clamping across +5V_SERVO & GND)| SS34 / B340A | SMA / DO-214AC | 1 | ₹5.00 | Local passives / Robu |
| 14 | J_SERVO | 3-Pin JST-XH / Berg Header (To MG90S Pan Servo: GND, +5V_SERVO, PWM) | B3B-XH-A(LF)(SN) | 2.54mm Pitch Male Right Angle | 1 | ₹8.00 | Robu / Chandni market |
| **D. ANALOG MIC PRE-AMP & I2S ADC (SYNCHRONOUS CLOCKING)** | | | | | | | |
| 15 | J_MIC_L, J_MIC_R| 2 × 2-Pin JST-PH Connectors (Left & Right Electret Microphone Capsule Inputs) | S2B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 2 | ₹7.00 (₹14.00 total) | Robu / Quartz Components |
| 16 | MIC_CAP | Electret Condenser Microphone Capsules (6mm or 9mm omnidirectional) | C9767 / KECG2742 | 2-Pin Lead Cylindrical (External) | 2 | ₹9.00 (₹18.00 total) | Supreme Electronics / Chandni market |
| 17 | U_PREAMP | Low-Noise Dual Rail-to-Rail Op-Amp (Microphone Pre-Amplifier) | LMV358IDR / OPA1678IDR | SOIC-8 | 1 | ₹18.00 | Evelta / Mouser / Chandni passives |
| 18 | U_ADC | 24-Bit 96kHz Stereo Audio A/D Converter (I2S Output, Single-Ended Input) | PCM1808PWR | TSSOP-14 | 1 | ₹110.00 | Evelta / Mouser India / DigiKey |
| 19 | R_FMT | 10kΩ 1% Resistor (PCM1808 Pin 9 `FMT` Pulldown to GND → 24-bit I2S Format) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 20 | R_MD0, R_MD1 | 2 × 10kΩ 1% Resistors (PCM1808 Pins 8/7 `MD0`/`MD1` Pulldowns to GND → Slave Mode) | RC0603FR-0710KL | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 21 | R_BYPAS | 10kΩ 1% Resistor (PCM1808 Pin 10 `BYPAS` Pulldown to GND → Enable DC HPF) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 22 | R_BIAS1, R_BIAS2 | 2.2kΩ 1% Resistors (Microphone JFET Bias Pull-up to 3.3V) | RC0603FR-072K2L | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 23 | R_FB1, R_FB2 | 100kΩ 1% Resistors (Op-amp Feedback Resistor / Gain Setting ~30dB) | RC0603FR-07100KL | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 24 | R_IN1, R_IN2 | 2.2kΩ 1% Resistors (Op-amp Inverting Input Resistors) | RC0603FR-072K2L | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 25 | C_IN1, C_IN2 | 1µF 16V X7R Ceramic AC Coupling Capacitors | CC0603KRX7R7BB105 | 0603 SMD | 2 | ₹2.50 (₹5.00 total) | Local passives |
| 26 | C_VREF | 10µF 10V X5R Ceramic VREF Decoupling Capacitor (PCM1808 Pin 5) | CL10A106KP8NNNC | 0603 SMD | 1 | ₹3.50 | Local passives |
| **E. AUDIO OUTPUT (SPEAKER AMPLIFIER)** | | | | | | | |
| 27 | U_AMP | 3.2W Mono Class-D I2S Audio Amplifier IC with Internal DAC | MAX98357AETE+T | 16-TQFN (3 × 3 mm, 0.5mm pitch) | 1 | ₹125.00 | Evelta / Mouser India / DigiKey |
| 28 | R_GAIN | **100kΩ 1% Resistor (Connected between Pin 2 `GAIN_SLOT` and `GND` → Fixed +12dB)** | RC0603FR-07100KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 29 | L_FLT1, L_FLT2 | Ferrite Beads (100MHz 600Ω, 2A rated for Speaker EMI filtering) | BLM18PG601SN1D | 0603 SMD | 2 | ₹6.00 (₹12.00 total) | Mouser / Evelta |
| 30 | C_FLT1, C_FLT2 | 220pF 50V C0G Ceramic Filter Capacitors | CC0603JRNPO9BN221 | 0603 SMD | 2 | ₹2.00 (₹4.00 total) | Local passives |
| 31 | C_AMP | 10µF 16V X5R Ceramic Bulk Capacitor (MAX98357A VDD supply) | CL10A106KP8NNNC | 0603 SMD | 1 | ₹3.50 | Local passives |
| 32 | J_SPK | 2-Pin JST-PH Connector (To 3W 4Ω/8Ω Internal Acoustic Speaker) | S2B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹7.00 | Robu / Quartz Components |
| **F. PERIPHERALS & SHIELDED NECK INTERFACE** | | | | | | | |
| 33 | J_RADAR | 3-Pin JST-PH Connector (To RCWL-0516 Microwave Radar: GND, +5V, OUT) | S3B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹7.00 | Robu / Quartz Components |
| 34 | J_NECK | 4-Pin JST-PH Shrouded Connector (To ESP32-S3 Head PCB: 5V, GND, TX, RX) | S4B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹9.00 | Robu / Quartz Components |
| 35 | CBL_NECK | 4-Core Shielded High-Flex Twisted Cable (28 AWG, Silicone, Braided Shield to GND)| CBL-4C-SHIELD-28AWG | External Cable Harness (30cm) | 1 | ₹35.00 | Evelta / Chandni wire market |
| **G. STATUS & TEST POINTS** | | | | | | | |
| 36 | LED1 | Green LED (5V System Power Indicator) | 0603-LED-GREEN | 0603 SMD | 1 | ₹2.00 | Local passives |
| 37 | R_LED1 | 1.0kΩ 1% Resistor (LED Current Limiter) | RC0603FR-071KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 38 | TP1–TP8 | SMD Test Points (GND, 5V, 3V3, BCLK, WS, DIN, DOUT, PWM) | RC0805 Surface Pad | 0805 Copper Pad | 8 | ₹0.00 | PCB Etched Copper |

---

## 3. Cost Summary (BOM 1)

| Subsystem | Components Included | Estimated Cost (INR ₹) |
| :--- | :--- | :---: |
| **Compute Subsystem** | Raspberry Pi Zero 2 W + SMD Headers | ₹2,185.00 |
| **Power & Primary Protection** | USB-C receptacle, Primary 3.0A PTC, P-MOSFET, 2200µF capacitor, TVS diode | ₹79.50 |
| **Servo Point-of-Load Protection** | Dedicated 1.5A PTC, 1000µF low-ESR cap, SS34 flyback diode, JST-XH header | ₹36.00 |
| **Audio Input (Synchronous Preamp + ADC)**| 2× Electret Mics, LMV358, PCM1808, strapping resistors, JSTs, passives | ₹172.00 |
| **Audio Output (Speaker Amp)** | MAX98357A IC, R_GAIN to GND, ferrite beads, bypass caps, JST speaker connector | ₹153.00 |
| **Peripherals & Shielded Neck Cable** | Radar JST-PH, Neck JST-PH 4-pin, 4-core shielded harness, status LED | ₹54.50 |
| **Bare PCB Fabrication** | 2-Layer FR4 (100×100mm batch at JLCPCB / Indian PCB fab @ 5-10 pcs) | ~₹150.00 |
| **Total Estimated Hardware Cost per Unit (BOM 1):** | | **₹2,830.00** |

---

## 4. Audio Clock Topology & Strapping Architecture

### 4.1 Master/Slave Clock Relationship
To prevent audio buffer overruns/underruns, pops, and clicks, **two independent, unsynchronized clock domains must never be allowed on the I2S bus**:
- **I2S Master:** The **Raspberry Pi Zero 2 W** acts as the definitive bus master.
- **I2S Slaves:** Both the **TI PCM1808 ADC** and the **Maxim MAX98357A DAC/Amp** act as synchronous I2S slaves.
- **Clock Distribution:**
  - `I2S_BCLK` (Pi GPIO18) drives PCM1808 Pin 12 (`BCK`) and MAX98357A Pin 14 (`BCLK`).
  - `I2S_WS` (Pi GPIO19) drives PCM1808 Pin 11 (`LRCK`) and MAX98357A Pin 15 (`LRCLK`).
  - `SCKI` (PCM1808 Pin 6 System Clock): Driven synchronously by Raspberry Pi **`GPCLK0` (GPIO4)** configured for 12.288 MHz (256 × 48 kHz). This ensures the PCM1808's internal delta-sigma modulator is frequency-locked to the Pi's audio DMA engine.

### 4.2 PCM1808 Hardware Strapping Configuration
| Pin # | Pin Name | Hardware Strapping Connection | State | Operating Mode Selected |
| :---: | :--- | :--- | :---: | :--- |
| **7** | `MD1` | Pulled to `GND` via 10kΩ `R_MD1` | `0` | **Slave Mode (256 fS)** |
| **8** | `MD0` | Pulled to `GND` via 10kΩ `R_MD0` | `0` | **Slave Mode (256 fS)** |
| **9** | `FMT` | Pulled to `GND` via 10kΩ `R_FMT` | `0` | **Standard 24-Bit I2S Format** |
| **10**| `BYPAS` | Pulled to `GND` via 10kΩ `R_BYPAS` | `0` | **DC-Blocking High-Pass Filter Active** |

---

## 5. Schematic Netlist & Pin Routing Table

| Pi Header Pin | BCM GPIO | Net Name | Connected To | Function / Description |
| :---: | :---: | :--- | :--- | :--- |
| Pin 2, 4 | — | `+5V_SYS` | USB-C 5V rail (via F1 3.0A PTC & Q1 P-MOSFET) | Primary 5V Power Supply into Pi |
| Pin 6, 9, 14, 20 | — | `GND` | System Ground Plane | Common Reference Ground |
| Pin 1 | — | `+3V3_PI` | LMV358 & PCM1808 VDD, Mic Bias Pullups | 3.3V Clean Digital/Analog Supply |
| Pin 7 | GPIO4 | `I2S_MCLK` | PCM1808 Pin 6 (`SCKI`) | Synchronous System Master Clock from Pi GPCLK0 |
| Pin 8 | GPIO14 | `BODY_UART_TX` | J_NECK Pin 3 (Head RX) | Serial Communication to ESP32-S3 (921600 baud) |
| Pin 10 | GPIO15 | `BODY_UART_RX` | J_NECK Pin 4 (Head TX) | Serial Communication from ESP32-S3 (921600 baud) |
| Pin 12 | GPIO18 | `I2S_BCLK` | PCM1808 Pin 12 (BCK) & MAX98357A Pin 14 (BCLK) | Shared I2S Bit Clock (Pi Master) |
| Pin 35 | GPIO19 | `I2S_WS` | PCM1808 Pin 11 (LRCK) & MAX98357A Pin 15 (LRCLK) | Shared I2S Word Select / Frame Clock |
| Pin 38 | GPIO20 | `I2S_DIN` | PCM1808 Pin 13 (DOUT) | Digital Audio Input from Microphones to Pi |
| Pin 40 | GPIO21 | `I2S_DOUT` | MAX98357A Pin 1 (DIN) | Digital Audio Output from Pi to Speaker Amp |
| Pin 32 | GPIO12 | `SERVO_PWM` | J_SERVO Pin 3 | Hardware PWM control for MG90S Pan Servo |
| Pin 36 | GPIO16 | `RADAR_IN` | J_RADAR Pin 3 (OUT) | Digital Logic Input from RCWL-0516 Radar |
| — | — | `MAX_GAIN` | MAX98357A Pin 2 to GND via 100kΩ R_GAIN | Fixed +12dB Audio Gain Setting |

---

## 6. Layout & Point-of-Load Servo Isolation Guidelines

1. **Dedicated Servo Isolation Branch:**
   - Place `F_SERVO` (1.5A polyfuse), `C_SERVO_LOC` (1000µF radial capacitor), and `D_SERVO` (SS34 Schottky diode) **directly beside `J_SERVO`**.
   - The diode cathode connects to `+5V_SERVO` and anode to `GND`. This immediately snubs inductive flyback spikes generated when the servo motor windings de-energize.
   - The local 1000µF capacitor supplies the instantaneous 800mA stall/start demand, preventing transient IR-drop along the PCB traces from dipping the Pi's 5V rail.
2. **Audio Ground & Shielding:**
   - Keep the analog microphone front-end (LMV358) and PCM1808 partitioned over a dedicated analog ground region linked back to the system star ground at the power entry.
   - The outer braided shield of `CBL_NECK` must be soldered to **Body GND only** (single-point shield grounding) to prevent ground loops between the body and head.
