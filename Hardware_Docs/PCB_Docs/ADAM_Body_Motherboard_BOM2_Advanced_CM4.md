# ADAM Body Motherboard — Production Bill of Materials (BOM 2)
## Option 2: Raspberry Pi Compute Module 4 (CM4 Lite) + All-Chip Discrete Peripheral ICs
### Advanced Computer Vision, OpenCV & Real-Time AI Platform

**Company:** Dgen Technologies Pvt. Ltd. (Kolkata, India)  
**Product:** ADAM — Autonomous Desktop AI Module  
**Board:** Body Shell Motherboard (BOM 2 — Advanced / High-Performance Vision)  
**Compute Core:** Raspberry Pi Compute Module 4 (CM4 Lite: 2GB or 4GB LPDDR4 RAM, Wi-Fi/BT)  
**Board-to-Board Mating:** Dual 100-Pin Hirose DF40 High-Density SMT Receptacles  
**PCB Layers:** 4-Layer FR4 (1.6mm thickness, 1oz outer / 1oz inner copper, 50Ω/90Ω controlled impedance)  
**Revision:** V2.2 (Hardware & Reliability Hardened)  
**Date:** 13 September 2026  
**Local Sourcing Hub:** Chandni Chowk / Princep St., Kolkata & Authorized Indian Distributors (Robu.in, Quartz Components, Evelta, Mouser India)

---

## 1. System Overview & Architecture

BOM 2 is designed for high-performance applications requiring real-time computer vision, OpenCV face/emotion tracking, frame processing, and multi-threaded Gemini Live execution:
- **Compute Core:** Broadcom **BCM2711 (Quad Cortex-A72 @ 1.5 GHz)** with **2GB (SKU: `CM4102000`) or 4GB (SKU: `CM4104000`) LPDDR4-3200 SDRAM** and onboard dual-band 2.4/5.0GHz Wi-Fi and Bluetooth 5.0 BLE.
- **RF Antenna Subsystem:** The wireless CM4 features an onboard U.FL (IPEX MHF1) connector; the carrier board BOM integrates the **Raspberry Pi Official Dual-Band Antenna Kit (MPN: `SC0288`)** to guarantee reliable Wi-Fi/BT link through the ADAM robot shell.
- **Form Factor:** CM4 Lite module (55 mm × 40 mm × 4.7 mm) snaps flat onto the custom motherboard via dual **Hirose DF40** 100-pin connectors (`DF40C-100DS-0.4V`). Zero bulky through-hole ports.
- **Storage Subsystem:** High-speed onboard **MicroSD card socket** on the custom carrier motherboard connected to the CM4's primary SDIO bus for the operating system and model weights.
- **USB Recovery & Reflash Interface:** Includes `nRPIBOOT` jumper/test pads and `GLOBAL_EN` reset pads allowing full eMMC flashing or system recovery via `rpiboot` over USB.
- **Dual-Stage Power & Servo Inrush Protection:** Direct 5V USB-C power entry with 3.5A primary polyfuse, AO3401A P-MOSFET, **2200µF 16V input reservoir capacitor**, plus a **dedicated local 1000µF 16V low-ESR capacitor, SS34 flyback Schottky diode, and 1.5A isolation polyfuse directly at `J_SERVO`** to protect the CM4 PMIC from motor brownouts.
- **Synchronous Audio ADC Subsystem:** Dual analog condenser microphone inputs with discrete 3.3V bias, **TI LMV358** dual op-amp preamplifier, and **TI PCM1808** 24-bit 96kHz stereo I2S ADC. Configured in **I2S Slave Mode** with explicit strapping resistors (`FMT=GND`, `MD0=GND`, `MD1=GND`, `BYPAS=GND`) and synchronous master clock driven by CM4 **`GPCLK0` (GPIO4)**.
- **Audio Output:** Discrete **Maxim MAX98357AETE+T** 3.2W Class-D I2S amplifier IC with **`GAIN_SLOT` explicitly tied to `GND` via a 100kΩ resistor (+12dB fixed gain)**.
- **Shielded Neck Link:** 4-pin locking JST-PH connector utilizing high-flex twisted shielded cabling (28 AWG) for reliable 921,600 baud communication with the ESP32-S3 Head Motherboard.

---

## 2. Complete Bill of Materials (BOM Table)

| Item # | Designator | Component / Description | Manufacturer Part Number (MPN) | Package / Footprint | Qty | Unit Price (INR ₹) | Primary Source (India / Kolkata) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **A. COMPUTE, ANTENNA & RECOVERY** | | | | | | | |
| 1 | U1 | Raspberry Pi Compute Module 4 Lite (2GB RAM, Wi-Fi, BCM2711 Quad A72) *(or 4GB variant)* | CM4102000 *(or CM4104000)* | Module (55 × 40 mm) | 1 | ₹3,850.00 *(2GB)* / ₹4,850.00 *(4GB)* | Robu.in / Supreme Electronics (Chandni Chowk indent) |
| 2 | ANT1 | Raspberry Pi Official External Dual-Band 2.4/5.0GHz Antenna Kit (U.FL to SMA + whip) | SC0288 / K210-ANT | U.FL / IPEX MHF1 Cable + Antenna | 1 | ₹420.00 | Robu.in / Silverline Electronics |
| 3 | J_CM4A, J_CM4B | 100-Pin 0.4mm Pitch Dual-Row Board-to-Board Receptacle (Mates with CM4) | DF40C-100DS-0.4V(51) | SMD Dual Row 100-Pin (Hirose DF40) | 2 | ₹240.00 (₹120 ea) | Mouser India / Evelta / Robu |
| 4 | J_SD | MicroSD Card Socket (Push-Pull, Reverse Mount SMD, 8-Pin + Card Detect) | 47219-2001 / DM3AT-SF-PEJM5 | SMD MicroSD Socket | 1 | ₹32.00 | Quartz Components / Robu / Mouser |
| 5 | J_RPIBOOT | 2-Pin 2.54mm Jumper / Test Pads (Short to GND for USB boot flashing via `rpiboot`) | 1X2-PIN-HEADER-2.54 | 2.54mm Pitch Male | 1 | ₹2.00 | Local passives / Robu |
| 6 | R_RPIBOOT | 10kΩ 1% Resistor (`nRPIBOOT` Pull-Up to 3.3V) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 7 | U_3V3 | High-Current Low-Dropout Linear Regulator (5V to 3.3V @ 1.5A for SDIO & ADC) | AMS1117-3.3 / AP2114H-3.3 | SOT-223 / SOT-89 | 1 | ₹12.00 | Local passives trader (Chandni Chowk) |
| **B. POWER INPUT & PRIMARY PROTECTION** | | | | | | | |
| 8 | J_PWR | USB Type-C Receptacle 16-Pin / 6-Pin (Power Only, 5V @ 3A–5A rated) | TYPE-C-16P-SMD | SMD Hybrid 16-Pin | 1 | ₹18.00 | Evelta / Robu / Chandni Chowk |
| 9 | R1, R2 | 5.1kΩ 1% 1/10W Resistors (USB Type-C CC1 / CC2 Pulldowns) | RC0603FR-075K1L | 0603 SMD | 2 | ₹1.50 | Local passives trader / Robu |
| 10 | F1 | SMD Resettable PTC Polyfuse (3.5A Hold, 4.0A–7.0A Trip, 6V min) - Primary Power | SMD1812P350TF / 1812L350 | 1812 SMD | 1 | ₹18.00 | Evelta / Mouser India |
| 11 | Q1 | P-Channel MOSFET (30V, 5.8A, Rds(on) < 35mΩ) Reverse Polarity Protection | AO3401A / SI2301CDS | SOT-23-3 | 1 | ₹6.00 | Quartz / Evelta / Chandni Chowk |
| 12 | D1 | TVS Diode Array (ESD Protection for USB & VBUS, 5V unidirectional) | USBLC6-2SC6 / SMBJ5.0A | SOT-23-6 / SMB | 1 | ₹12.00 | Evelta / Mouser India |
| 13 | C_BULK_IN | **2200µF 16V Low-ESR Radial Electrolytic Capacitor** (Input Power Reservoir) | UPW1C222MHD / EEU-FR1C222 | Radial (10×20mm, 5mm pitch) | 1 | ₹24.00 | Supreme Electronics / Classic Electronics (Chandni) |
| 14 | C1–C5 | 10µF 16V X5R Ceramic Decoupling Capacitors (Power & CM4 rails) | CL10A106KP8NNNC | 0603 SMD | 5 | ₹3.50 | Local passives / Robu |
| 15 | C6–C10 | 100nF (0.1µF) 50V X7R Ceramic Decoupling Capacitors | CC0603KRX7R9BB104 | 0603 SMD | 5 | ₹1.50 | Local passives / Robu |
| **C. SERVO POINT-OF-LOAD INRUSH & BACK-EMF PROTECTION** | | | | | | | |
| 16 | F_SERVO | SMD Resettable PTC Polyfuse (0.75A Hold, 1.5A Trip) - Servo Subsystem Isolation | SMD1206P075TF | 1206 SMD | 1 | ₹9.00 | Evelta / Mouser India |
| 17 | C_SERVO_LOC| **1000µF 16V Low-ESR Radial Electrolytic Capacitor** (Mounted at J_SERVO) | UPW1C102MPD / EEU-FR1C102 | Radial (8×15mm, 3.5mm pitch) | 1 | ₹14.00 | Supreme Electronics / Chandni market |
| 18 | D_SERVO | 3A 40V Schottky Barrier Diode (Flyback / Back-EMF Clamping across +5V_SERVO & GND)| SS34 / B340A | SMA / DO-214AC | 1 | ₹5.00 | Local passives / Robu |
| 19 | J_SERVO | 3-Pin JST-XH / Berg Header (To MG90S Pan Servo: GND, +5V_SERVO, PWM) | B3B-XH-A(LF)(SN) | 2.54mm Pitch Male Right Angle | 1 | ₹8.00 | Robu / Chandni market |
| **D. ANALOG MIC PRE-AMP & I2S ADC (SYNCHRONOUS CLOCKING)** | | | | | | | |
| 20 | J_MIC_L, J_MIC_R| 2 × 2-Pin JST-PH Connectors (Left & Right Electret Microphone Capsule Inputs) | S2B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 2 | ₹7.00 (₹14.00 total) | Robu / Quartz Components |
| 21 | MIC_CAP | Electret Condenser Microphone Capsules (6mm or 9mm omnidirectional) | C9767 / KECG2742 | 2-Pin Lead Cylindrical (External) | 2 | ₹9.00 (₹18.00 total) | Supreme Electronics / Chandni market |
| 22 | U_PREAMP | Low-Noise Dual Rail-to-Rail Op-Amp (Microphone Pre-Amplifier) | LMV358IDR / OPA1678IDR | SOIC-8 | 1 | ₹18.00 | Evelta / Mouser / Chandni passives |
| 23 | U_ADC | 24-Bit 96kHz Stereo Audio A/D Converter (I2S Output, Single-Ended Input) | PCM1808PWR | TSSOP-14 | 1 | ₹110.00 | Evelta / Mouser India / DigiKey |
| 24 | R_FMT | 10kΩ 1% Resistor (PCM1808 Pin 9 `FMT` Pulldown to GND → 24-bit I2S Format) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 25 | R_MD0, R_MD1 | 2 × 10kΩ 1% Resistors (PCM1808 Pins 8/7 `MD0`/`MD1` Pulldowns to GND → Slave Mode) | RC0603FR-0710KL | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 26 | R_BYPAS | 10kΩ 1% Resistor (PCM1808 Pin 10 `BYPAS` Pulldown to GND → Enable DC HPF) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 27 | R_BIAS1, R_BIAS2 | 2.2kΩ 1% Resistors (Microphone JFET Bias Pull-up to 3.3V) | RC0603FR-072K2L | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 28 | R_FB1, R_FB2 | 100kΩ 1% Resistors (Op-amp Feedback Resistor / Gain Setting ~30dB) | RC0603FR-07100KL | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 29 | R_IN1, R_IN2 | 2.2kΩ 1% Resistors (Op-amp Inverting Input Resistors) | RC0603FR-072K2L | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 30 | C_IN1, C_IN2 | 1µF 16V X7R Ceramic AC Coupling Capacitors | CC0603KRX7R7BB105 | 0603 SMD | 2 | ₹2.50 (₹5.00 total) | Local passives |
| 31 | C_VREF | 10µF 10V X5R Ceramic VREF Decoupling Capacitor (PCM1808 Pin 5) | CL10A106KP8NNNC | 0603 SMD | 1 | ₹3.50 | Local passives |
| **E. AUDIO OUTPUT (SPEAKER AMPLIFIER)** | | | | | | | |
| 32 | U_AMP | 3.2W Mono Class-D I2S Audio Amplifier IC with Internal DAC | MAX98357AETE+T | 16-TQFN (3 × 3 mm, 0.5mm pitch) | 1 | ₹125.00 | Evelta / Mouser India / DigiKey |
| 33 | R_GAIN | **100kΩ 1% Resistor (Connected between Pin 2 `GAIN_SLOT` and `GND` → Fixed +12dB)** | RC0603FR-07100KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 34 | L_FLT1, L_FLT2 | Ferrite Beads (100MHz 600Ω, 2A rated for Speaker EMI filtering) | BLM18PG601SN1D | 0603 SMD | 2 | ₹6.00 (₹12.00 total) | Mouser / Evelta |
| 35 | C_FLT1, C_FLT2 | 220pF 50V C0G Ceramic Filter Capacitors | CC0603JRNPO9BN221 | 0603 SMD | 2 | ₹2.00 (₹4.00 total) | Local passives |
| 36 | C_AMP | 10µF 16V X5R Ceramic Bulk Capacitor (MAX98357A VDD supply) | CL10A106KP8NNNC | 0603 SMD | 1 | ₹3.50 | Local passives |
| 37 | J_SPK | 2-Pin JST-PH Connector (To 3W 4Ω/8Ω Internal Acoustic Speaker) | S2B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹7.00 | Robu / Quartz Components |
| **F. PERIPHERALS & SHIELDED NECK INTERFACE** | | | | | | | |
| 38 | J_RADAR | 3-Pin JST-PH Connector (To RCWL-0516 Microwave Radar: GND, +5V, OUT) | S3B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹7.00 | Robu / Quartz Components |
| 39 | J_NECK | 4-Pin JST-PH Shrouded Connector (To ESP32-S3 Head PCB: 5V, GND, TX, RX) | S4B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹9.00 | Robu / Quartz Components |
| 40 | CBL_NECK | 4-Core Shielded High-Flex Twisted Cable (28 AWG, Silicone, Braided Shield to GND)| CBL-4C-SHIELD-28AWG | External Cable Harness (30cm) | 1 | ₹35.00 | Evelta / Chandni wire market |
| **G. STATUS & TEST POINTS** | | | | | | | |
| 41 | LED1, LED2 | Green / Amber LEDs (5V Power & CM4 ACT Activity) | 0603-LED | 0603 SMD | 2 | ₹4.00 (₹2 ea) | Local passives |
| 42 | R_LED1, R_LED2 | 1.0kΩ 1% Resistors (LED Current Limiters) | RC0603FR-071KL | 0603 SMD | 2 | ₹3.00 | Local passives |
| 43 | TP1–TP10 | SMD Test Points (GND, 5V, 3V3, BCLK, WS, DIN, DOUT, PWM, GLOBAL_EN, RPIBOOT)| RC0805 Surface Pad | 0805 Copper Pad | 10 | ₹0.00 | PCB Etched Copper |

---

## 3. Cost Summary (BOM 2)

| Subsystem | Components Included | Estimated Cost (INR ₹) |
| :--- | :--- | :---: |
| **Compute Subsystem** | Raspberry Pi CM4 Lite (2GB Wi-Fi) + Dual Hirose DF40 Receptacles | ₹4,090.00 |
| **Antenna Subsystem** | Raspberry Pi Official Dual-Band Antenna Kit (`SC0288`) with U.FL pigtail | ₹420.00 |
| **Storage & 3.3V Power** | MicroSD socket + AMS1117-3.3 LDO + filter caps + nRPIBOOT recovery header | ₹57.50 |
| **Power & Primary Protection** | USB-C receptacle, Primary 3.5A PTC, P-MOSFET, 2200µF capacitor, TVS diode | ₹81.50 |
| **Servo Point-of-Load Protection** | Dedicated 1.5A PTC, 1000µF low-ESR cap, SS34 flyback diode, JST-XH header | ₹36.00 |
| **Audio Input (Synchronous Preamp + ADC)**| 2× Electret Mics, LMV358, PCM1808, strapping resistors, JSTs, passives | ₹172.00 |
| **Audio Output (Speaker Amp)** | MAX98357A IC, R_GAIN to GND, ferrite beads, bypass caps, JST speaker connector | ₹153.00 |
| **Peripherals & Shielded Neck Cable** | Radar JST-PH, Neck JST-PH 4-pin, 4-core shielded harness, status LEDs | ₹57.50 |
| **Bare PCB Fabrication** | 4-Layer FR4 (100×100mm batch at JLCPCB / Indian PCB fab @ 5-10 pcs) | ~₹350.00 |
| **Total Estimated Hardware Cost per Unit (BOM 2):** | | **₹5,417.50** *(2GB RAM)* |

*(Note: Upgrading to the 4GB RAM CM4 variant adds ₹1,000 to the compute cost, bringing total unit BOM to ~₹6,417.50).*

---

## 4. Audio Clock Topology & Strapping Architecture

### 4.1 Master/Slave Clock Relationship
- **I2S Master:** The **Raspberry Pi Compute Module 4** acts as bus master.
- **I2S Slaves:** Both the **TI PCM1808 ADC** and the **Maxim MAX98357A DAC/Amp** act as synchronous I2S slaves.
- **Clock Distribution:**
  - `I2S_BCLK` (CM4 GPIO18 on `J_CM4B` Pin 34) drives PCM1808 Pin 12 (`BCK`) and MAX98357A Pin 14 (`BCLK`).
  - `I2S_WS` (CM4 GPIO19 on `J_CM4B` Pin 36) drives PCM1808 Pin 11 (`LRCK`) and MAX98357A Pin 15 (`LRCLK`).
  - `SCKI` (PCM1808 Pin 6 System Clock): Driven synchronously by CM4 **`GPCLK0` (GPIO4 on `J_CM4B` Pin 10)** configured for 12.288 MHz (256 × 48 kHz). Completely prevents buffer drift, pops, and clicks.

### 4.2 PCM1808 Hardware Strapping Configuration
| Pin # | Pin Name | Hardware Strapping Connection | State | Operating Mode Selected |
| :---: | :--- | :--- | :---: | :--- |
| **7** | `MD1` | Pulled to `GND` via 10kΩ `R_MD1` | `0` | **Slave Mode (256 fS)** |
| **8** | `MD0` | Pulled to `GND` via 10kΩ `R_MD0` | `0` | **Slave Mode (256 fS)** |
| **9** | `FMT` | Pulled to `GND` via 10kΩ `R_FMT` | `0` | **Standard 24-Bit I2S Format** |
| **10**| `BYPAS` | Pulled to `GND` via 10kΩ `R_BYPAS` | `0` | **DC-Blocking High-Pass Filter Active** |

---

## 5. Hirose DF40 Pinout & Carrier Board Routing Table

### 5.1 Power, Ground & Recovery Nets — J_CM4A
| Hirose Pin | Net Name | Connected To | Function / Description |
| :---: | :--- | :--- | :--- |
| Pins 86, 88, 90, 92, 94, 96, 98, 100 | `+5V_SYS` | Main 5V Rail (via F1 & Q1) | Primary 5V Power Supply into CM4 |
| Pins 1, 4, 7, 10, 13, 16, 19, etc. | `GND` | System Ground Plane | Common Reference Ground |
| J_CM4A Pin 93 | `nRPIBOOT` | J_RPIBOOT & 10k Pullup to 3.3V | Pull LOW to force USB eMMC recovery boot |
| J_CM4A Pin 99 | `GLOBAL_EN` | TP_EN Test Pad | Hardware Reset / Run pin |

### 5.2 MicroSD Card Interface (SDIO) — J_CM4A to J_SD
| Hirose DF40 Pin | Net Name | MicroSD Pin | Function / Description |
| :---: | :--- | :---: | :--- |
| J_CM4A Pin 67 | `SD_DATA0` | Pin 7 (DAT0) | Bidirectional Data Line 0 |
| J_CM4A Pin 69 | `SD_DATA1` | Pin 8 (DAT1) | Bidirectional Data Line 1 |
| J_CM4A Pin 71 | `SD_DATA2` | Pin 1 (DAT2) | Bidirectional Data Line 2 |
| J_CM4A Pin 73 | `SD_DATA3` | Pin 2 (CD/DAT3) | Bidirectional Data Line 3 |
| J_CM4A Pin 75 | `SD_CLK` | Pin 5 (CLK) | SDIO Clock (Length-matched to ±1mm) |
| J_CM4A Pin 77 | `SD_CMD` | Pin 3 (CMD) | SDIO Command Line |
| Onboard 3.3V | `+3V3_SD` | Pin 4 (VDD) | 3.3V Supply with 10µF + 100nF decoupling |

### 5.3 Audio, UART & Peripheral GPIO Mapping — J_CM4B
| Hirose DF40 Pin | BCM GPIO | Net Name | Connected To | Function / Description |
| :---: | :---: | :--- | :--- | :--- |
| J_CM4B Pin 10 | GPIO4 | `I2S_MCLK` | PCM1808 Pin 6 (`SCKI`) | Synchronous Master Clock from CM4 GPCLK0 |
| J_CM4B Pin 26 | GPIO14 | `BODY_UART_TX` | J_NECK Pin 3 (Head RX) | Serial Output to ESP32-S3 Head (921600 baud) |
| J_CM4B Pin 28 | GPIO15 | `BODY_UART_RX` | J_NECK Pin 4 (Head TX) | Serial Input from ESP32-S3 Head (921600 baud) |
| J_CM4B Pin 34 | GPIO18 | `I2S_BCLK` | PCM1808 Pin 12 & MAX98357A Pin 14 | Shared I2S Bit Clock (CM4 Master) |
| J_CM4B Pin 36 | GPIO19 | `I2S_WS` | PCM1808 Pin 11 & MAX98357A Pin 15 | Shared I2S Word Select / Frame Clock |
| J_CM4B Pin 38 | GPIO20 | `I2S_DIN` | PCM1808 Pin 13 (DOUT) | Digital Audio Input from Mics to CM4 |
| J_CM4B Pin 40 | GPIO21 | `I2S_DOUT` | MAX98357A Pin 1 (DIN) | Digital Audio Output from CM4 to Speaker Amp |
| J_CM4B Pin 22 | GPIO12 | `SERVO_PWM` | J_SERVO Pin 3 | Hardware PWM control for MG90S Pan Servo |
| J_CM4B Pin 30 | GPIO16 | `RADAR_IN` | J_RADAR Pin 3 (OUT) | Digital Logic Input from RCWL-0516 Radar |
| J_CM4B Pin 84 | — | `CM4_ACT_LED` | LED2 (via 1kΩ) | Active Low Activity LED |
| — | — | `MAX_GAIN` | MAX98357A Pin 2 to GND via 100kΩ R_GAIN | Fixed +12dB Audio Gain Setting |

---

## 6. Layout, Antenna & Point-of-Load Servo Isolation Guidelines

1. **U.FL Antenna Mechanical Routing:**
   - Reserve a mechanical clearance notch on the carrier board around the CM4's U.FL connector.
   - Route the flexible U.FL coaxial pigtail towards the rear of the ADAM body shell, terminating in an SMA bulkhead or adhesive dipole mounted on plastic (not shielded by internal metal or copper).
2. **Dedicated Servo Point-of-Load Isolation:**
   - Mount `F_SERVO` (1.5A polyfuse), `C_SERVO_LOC` (1000µF radial capacitor), and `D_SERVO` (SS34 Schottky diode) **directly beside `J_SERVO`**.
   - Clamps motor inductive kickback immediately and isolates motor acceleration spikes from the CM4's internal PMIC rail.
3. **4-Layer PCB Stack-up:**
   - **Layer 1 (Top):** High-speed SDIO, I2S audio, component pads.
   - **Layer 2 (Inner 1):** Solid GND Plane (unbroken ground return).
   - **Layer 3 (Inner 2):** Power Plane (+5V_SYS, +3.3V, +5V_SERVO).
   - **Layer 4 (Bottom):** Non-critical peripheral traces (UART, LEDs, Radar).
