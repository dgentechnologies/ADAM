# ADAM Head Motherboard — Production Bill of Materials (BOM)
## ESP32-S3 Unified Camera, Display, Touch & Tilt Controller

**Company:** Dgen Technologies Pvt. Ltd. (Kolkata, India)  
**Product:** ADAM — Autonomous Desktop AI Module  
**Board:** Head Motherboard (Unified ESP32-S3 Architecture)  
**Compute Core:** ESP32-S3-WROOM-1-N8R8 / N16R8 (8MB Octal PSRAM)  
**PCB Layers:** 2-Layer or 4-Layer FR4 (1.6mm thickness, 1oz copper)  
**Revision:** V1.2 (Hardware & Reliability Hardened)  
**Date:** 13 September 2026  
**Local Sourcing Hub:** Chandni Chowk / Princep St., Kolkata & Authorized Indian Distributors (Robu.in, Evelta, Quartz Components, Mouser India)

---

## 1. System Overview & Engineering Highlights

The ADAM Head Motherboard consolidates the legacy dual-board (ESP32-CAM + Raspberry Pi Pico) head electronics into a **single integrated custom PCB**:
- **ESP32-S3 Compute & Memory:** High-performance 240MHz dual-core Xtensa LX7 MCU with **8MB Octal PSRAM** for smooth camera frame buffering and 320×240 TFT animation rendering. **16MB Flash (`N16R8`)** is recommended to accommodate dual OTA application partitions + LittleFS storage for face animation frames.
- **Zero Strapping Pin Conflicts:** `TFT_DC` and `TFT_CS` are mapped to **`GPIO37` and `GPIO38`** (completely removing them from bootstrapping pins `GPIO45`/`GPIO46` to prevent `VDD_SPI` flash voltage locking). Camera SCCB is mapped to **`GPIO39`/`GPIO40`**, and Touch channels are mapped to **`GPIO1`, `GPIO2`, `GPIO4`, and `GPIO14`**, leaving strapping pin `GPIO3` completely free.
- **Camera Subsystem:** 24-Pin 0.5mm pitch FPC connector mating with an **OV2640 2MP** camera module (66mm/75mm extended flex with wide-angle lens for neck tilt articulation). Features a dedicated **10kΩ + 100nF RC power-on reset delay network** on the camera `RESET` line and tied-down `PWDN` for clean sensor initialization.
- **Display Subsystem:** 20-Pin 0.5mm pitch FPC connector mating with a **2.4" 320×240 ILI9341 color TFT**, driven via high-speed DMA SPI with fixed 3.3V backlight.
- **Dual-Mode Touch Interface:** Four 3-pin JST-PH connectors for external **TTP223** touch modules, routed to ESP32-S3 pins (`GPIO1`, `GPIO2`, `GPIO4`, `GPIO14`) that also feature **native capacitive touch hardware (TOUCH1, TOUCH2, TOUCH4, TOUCH14)** and onboard circular copper touch pads.
- **Tilt Actuation & Back-EMF Protection:** 3-pin header driving the **MG90S metal gear servo** directly from the 5V rail with a **220µF 16V low-ESR bulk capacitor** and an **SS14 Schottky flyback clamping diode** to snub inductive spikes.
- **Neck Link:** 4-pin locking JST-PH connector (5V, GND, TX, RX) communicating with the body Raspberry Pi at 921,600 baud over high-flex 28 AWG shielded twisted-pair cabling.
- **Pogo-Pin Programming Interface:** Circular copper pads on the PCB bottom for factory pogo-pin programming/debug (`3V3`, `GND`, `EN`, `IO0`, `TX0`, `RX0`, `USB_D-`, `USB_D+`), eliminating bulky USB ports.

---

## 2. Complete Bill of Materials (BOM Table)

| Item # | Designator | Component / Description | Manufacturer Part Number (MPN) | Package / Footprint | Qty | Unit Price (INR ₹) | Primary Source (India / Kolkata) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **A. COMPUTE & MEMORY CORE** | | | | | | | |
| 1 | U1 | ESP32-S3 Dual-Core MCU Module (16MB Flash, 8MB Octal PSRAM, PCB Antenna) *(or 8MB N8R8)* | ESP32-S3-WROOM-1-N16R8 *(or N8R8)* | SMD Module (18 × 25.5 mm, 41 castellated pads) | 1 | ₹449.00 *(N16R8)* / ₹399.00 *(N8R8)* | Robu.in / Evelta / Amar Radio (Chandni Chowk) |
| **B. POWER REGULATION & DECOUPLING** | | | | | | | |
| 2 | U2 | 3.3V 1.0A Low-Dropout Linear Voltage Regulator (5V to 3.3V) | AMS1117-3.3 / AP2114H-3.3TRG1 | SOT-223 | 1 | ₹6.50 | Supreme Electronics (Chandni) / Robu.in |
| 3 | C_SERVO | **220µF 16V Low-ESR Radial Electrolytic Capacitor** (Servo Inrush Buffer) | UPW1C221MPD / ECA-1CHG221 | Radial (6.3 × 11 mm, 2.5mm pitch) | 1 | ₹6.00 | Classic Electronics / Chandni market |
| 4 | D_SERVO | 1A 40V Schottky Barrier Diode (Flyback Clamping across Servo +5V & GND) | SS14 / 1N5819WS | SOD-123 / SMA | 1 | ₹3.50 | Local passives / Robu |
| 5 | C1, C2, C3 | 10µF 16V X5R Ceramic Capacitors (LDO Input, LDO Output, Camera 3.3V) | CL10A106KP8NNNC | 0603 SMD | 3 | ₹3.50 | Local passives trader / Robu |
| 6 | C4–C8 | 100nF (0.1µF) 50V X7R Ceramic Decoupling Capacitors | CC0603KRX7R9BB104 | 0603 SMD | 5 | ₹1.50 | Local passives / Robu |
| **C. CAMERA SUBSYSTEM (OV2640)** | | | | | | | |
| 7 | J_CAM | 24-Pin 0.5mm Pitch Bottom-Contact FPC SMD Connector (ZIF / Flip-Lock) | FH12-24S-0.5SH(55) / FPC05-24P | 0.5mm Pitch SMD R/A | 1 | ₹22.00 | Evelta / Robu.in / Chandni market lane |
| 8 | CAM_MOD | OV2640 2MP Camera Sensor Module (66mm/75mm Extended Flex, Wide Angle) | OV2640-FPC-75MM | 24-Pin 0.5mm FPC Ribbon | 1 | ₹260.00 | Robu.in / Supreme Electronics (Chandni) |
| 9 | R_SIOD, R_SIOC | 2 × 2.2kΩ 1% Resistors (Camera SCCB Pull-ups on GPIO39 & GPIO40) | RC0603FR-072K2L | 0603 SMD | 2 | ₹1.50 (₹3.00 total) | Local passives |
| 10 | R_CAM_RST | 10kΩ 1% Resistor (Camera RESET Pull-Up to 3.3V) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 11 | C_CAM_RST | 100nF 50V Ceramic Capacitor (Camera RESET Power-On Delay) | CC0603KRX7R9BB104 | 0603 SMD | 1 | ₹1.50 | Local passives |
| 12 | R_PWDN | 10kΩ 1% Resistor (Camera PWDN Pulldown to GND) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| **D. DISPLAY SUBSYSTEM (ILI9341 2.4" TFT)** | | | | | | | |
| 13 | J_TFT | 20-Pin 0.5mm Pitch Bottom-Contact FPC SMD Connector (Back-Flip Lock) | 20PIN-05-FPC / 544601 | 0.5mm Pitch SMD R/A | 1 | ₹15.00 | Robu.in / Evelta |
| 14 | TFT_PANEL | 2.4" 320×240 ILI9341 Color TFT LCD Panel (20-pin bare flex ribbon) | ILI9341-2.4-20P-FPC | 20-Pin FPC Bare Panel | 1 | ₹380.00 | Robu.in / Quartz Components / Chandni |
| 15 | R_BL | 10Ω 1% 1/8W Resistor (Backlight Current Limiter from 3.3V rail) | RC0805FR-0710RL | 0805 SMD | 1 | ₹2.00 | Local passives |
| **E. DUAL-MODE TOUCH SUBSYSTEM** | | | | | | | |
| 16 | J_T1–J_T4 | 4 × 3-Pin JST-PH Connectors (To external TTP223 modules: 3V3, GND, OUT) | S3B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 4 | ₹7.50 (₹30.00 total) | Evelta / Robu.in / Chandni market |
| 17 | TP_T1–TP_T4 | Circular Copper Touch Contact Pads (Native Capacitive Touch Testing) | PAD-TOUCH-ROUND-6MM | PCB Top Copper Surface Pad | 4 | ₹0.00 | PCB Etched Copper |
| 18 | TTP223_MOD | TTP223 Capacitive Touch Sensor Breakout Modules (Red/Blue mini modules) | TTP223-MODULE | Mini PCB Module (External) | 4 | ₹18.00 (₹72.00 total) | Supreme Electronics / Chandni Chowk |
| 19 | R_PD1–R_PD4 | 10kΩ 1% Resistors (Optional Pulldowns for Touch Inputs) | RC0603FR-0710KL | 0603 SMD | 4 | ₹1.50 (₹6.00 total) | Local passives |
| **F. TILT SERVO SUBSYSTEM** | | | | | | | |
| 20 | J_SERVO | 3-Pin 2.54mm Breakable Male Pin Header (To MG90S Tilt Servo: GND, 5V, PWM) | 1X3-PIN-HEADER-2.54 | 2.54mm Pitch Male Vertical | 1 | ₹2.00 | Supreme Electronics / Robu.in |
| 21 | SERVO_MOD | MG90S Metal Gear Micro Servo 9g (External Tilt Actuator) | MG90S-SERVO | External Actuator | 1 | ₹125.00 | Robu.in / Supreme Electronics |
| **G. NECK 4-WIRE SHIELDED INTERCONNECT** | | | | | | | |
| 22 | J_NECK | 4-Pin JST-PH Shrouded Connector (To Body Motherboard: 5V, GND, TX, RX) | S4B-PH-K-S(LF)(SN) | 2.0mm Pitch Male Right Angle | 1 | ₹9.00 | Robu.in / Evelta |
| **H. POGO-PIN PROGRAMMING & RESET** | | | | | | | |
| 23 | TP_POGO1–TP_POGO8 | Circular Copper Pogo-Pin Test Pads (3V3, GND, EN, IO0, TX0, RX0, DM, DP) | PAD-POGO-1.5MM | 1.5mm Gold/Tin Copper Pad | 8 | ₹0.00 | PCB Bottom Etched Copper |
| 24 | R_EN | 10kΩ 1% Resistor (ESP32-S3 EN Pin Pull-Up) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 25 | C_EN | 1µF 16V X7R Ceramic Capacitor (ESP32-S3 EN Power-On Delay) | CC0603KRX7R7BB105 | 0603 SMD | 1 | ₹2.50 | Local passives |
| 26 | R_BOOT | 10kΩ 1% Resistor (ESP32-S3 GPIO0 / Boot Pull-Up) | RC0603FR-0710KL | 0603 SMD | 1 | ₹1.50 | Local passives |
| 27 | LED1 | Green LED (3.3V Power Indicator) | 0603-LED-GREEN | 0603 SMD | 1 | ₹2.00 | Local passives |
| 28 | R_LED1 | 1.0kΩ 1% Resistor (LED Current Limiter) | RC0603FR-071KL | 0603 SMD | 1 | ₹1.50 | Local passives |

---

## 3. Cost Summary (Head Motherboard)

| Subsystem | Components Included | Estimated Cost (INR ₹) |
| :--- | :--- | :---: |
| **Compute & Memory Core** | ESP32-S3-WROOM-1-N16R8 Module (16MB Flash, 8MB PSRAM) | ₹449.00 |
| **Power Regulation & Servo Protection** | AMS1117-3.3 LDO, 220µF 16V cap, SS14 flyback diode, filter caps | ₹30.00 |
| **Camera Interface** | 24-Pin FPC Connector + OV2640 2MP Sensor (75mm flex) + RC reset network | ₹289.00 |
| **Display Interface** | 20-Pin FPC Connector + 2.4" ILI9341 TFT Panel + resistor | ₹397.00 |
| **Touch Subsystem** | 4× JST-PH Connectors + 4× TTP223 Modules + passives + native pads | ₹108.00 |
| **Servo & Neck Connectors** | 3-Pin Servo Header + MG90S Servo + 4-Pin Neck JST-PH | ₹136.00 |
| **Programming & Passives** | Pogo pads, EN/BOOT RC networks, Power LED | ₹9.00 |
| **Bare PCB Fabrication** | 2-Layer FR4 (60×50mm batch at JLCPCB / Indian fab @ 5-10 pcs) | ~₹120.00 |
| **Total Estimated Hardware Cost per Unit (Head):** | | **₹1,538.00** *(16MB)* / **₹1,488.00** *(8MB)* |

*(If using native capacitive touch pads instead of external TTP223 modules, deduct ₹72.00 from the touch subsystem, bringing total unit cost to **~₹1,466.00**).*

---

## 4. Conflict-Free ESP32-S3 Schematic Pin Routing Table

Every GPIO has been assigned to completely eliminate boot strapping risks:

| ESP32-S3 Pad | GPIO | Net Name | Connected To | Function / Description | Strapping Status |
| :---: | :---: | :--- | :--- | :--- | :---: |
| Pad 1 | GND | `GND` | Ground Plane | System Common Ground | — |
| Pad 2 | 3V3 | `+3V3_SYS` | AMS1117-3.3 Output & 10µF Cap | Main 3.3V Power Rail | — |
| Pad 3 | EN | `ESP_EN` | Pogo Pad TP_EN, R_EN (10k), C_EN (1µF) | Chip Enable / Hardware Reset | — |
| Pad 11 | GPIO1 | `TOUCH1` | J_T1 Pin 3 & Copper Touch Pad TP_T1 | Left Cheek Touch (TTP223 / Native TOUCH1) | Safe |
| Pad 12 | GPIO2 | `TOUCH2` | J_T2 Pin 3 & Copper Touch Pad TP_T2 | Right Cheek Touch (TTP223 / Native TOUCH2) | Safe |
| Pad 13 | GPIO3 | `NC` | **Unconnected / Reserved** | Boot Strap Pin (JTAG Select) — Left completely free! | **ISOLATED** |
| Pad 15 | GPIO4 | `TOUCH3` | J_T3 Pin 3 & Copper Touch Pad TP_T3 | Petting-A Touch (TTP223 / Native TOUCH4) | Safe |
| Pad 14 | GPIO14 | `TOUCH4` | J_T4 Pin 3 & Copper Touch Pad TP_T4 | Petting-B Touch (TTP223 / Native TOUCH14) | Safe |
| Pad 16 | GPIO5 | `NC` | Reserved / Free GPIO | Free GPIO (or optional 5th Touch TOUCH5) | Safe |
| Pad 17 | GPIO6 | `CAM_VSYNC`| J_CAM Pin 14 (VSYNC) | Camera Vertical Synchronization | Safe |
| Pad 18 | GPIO7 | `CAM_HREF` | J_CAM Pin 16 (HREF) | Camera Horizontal Reference | Safe |
| Pad 19 | GPIO8 | `CAM_D2` | J_CAM Pin 13 (D2) | Camera Pixel Data Bit 2 | Safe |
| Pad 20 | GPIO9 | `CAM_D1` | J_CAM Pin 15 (D1) | Camera Pixel Data Bit 1 | Safe |
| Pad 21 | GPIO10 | `CAM_D3` | J_CAM Pin 11 (D3) | Camera Pixel Data Bit 3 | Safe |
| Pad 22 | GPIO11 | `CAM_D0` | J_CAM Pin 17 (D0) | Camera Pixel Data Bit 0 | Safe |
| Pad 23 | GPIO12 | `CAM_D4` | J_CAM Pin 9 (D4) | Camera Pixel Data Bit 4 | Safe |
| Pad 24 | GPIO13 | `CAM_PCLK` | J_CAM Pin 18 (PCLK) | Camera Pixel Clock Input | Safe |
| Pad 25 | GPIO15 | `CAM_XCLK` | J_CAM Pin 12 (XCLK) | Master Clock to Camera (20MHz/24MHz) | Safe |
| Pad 26 | GPIO16 | `CAM_D7` | J_CAM Pin 3 (D7) | Camera Pixel Data Bit 7 | Safe |
| Pad 27 | GPIO17 | `CAM_D6` | J_CAM Pin 5 (D6) | Camera Pixel Data Bit 6 | Safe |
| Pad 28 | GPIO18 | `CAM_D5` | J_CAM Pin 7 (D5) | Camera Pixel Data Bit 5 | Safe |
| Pad 31 | GPIO35 | `TFT_MOSI` | J_TFT Pin 5 (SDI/MOSI) | Display SPI Data Output | Safe |
| Pad 32 | GPIO36 | `TFT_SCK` | J_TFT Pin 8 (SCK) | Display SPI Clock Output | Safe |
| Pad 33 | **GPIO37**| `TFT_DC` | J_TFT Pin 7 (DC) | **Display Data / Command Select (Moved off GPIO45!)** | **Safe** |
| Pad 34 | **GPIO38**| `TFT_CS` | J_TFT Pin 9 (CS) | **Display SPI Chip Select (Moved off GPIO46!)** | **Safe** |
| Pad 35 | **GPIO39**| `CAM_SIOD` | J_CAM Pin 10 & 2.2k Pullup | **Camera SCCB / I2C Serial Data (Moved off GPIO4!)** | **Safe** |
| Pad 36 | **GPIO40**| `CAM_SIOC` | J_CAM Pin 8 & 2.2k Pullup | **Camera SCCB / I2C Serial Clock (Moved off GPIO5!)** | **Safe** |
| Pad 37 | GPIO48 | `SERVO_PWM`| J_SERVO Pin 3 | LEDC PWM Signal to MG90S Tilt Servo | Safe |
| Pad 38 | GPIO43 | `HEAD_TX` | J_NECK Pin 4 & Pogo Pad TP_TX | UART0 Serial Output to Body Pi (921600 baud) | Safe |
| Pad 39 | GPIO44 | `HEAD_RX` | J_NECK Pin 3 & Pogo Pad TP_RX | Serial Input from Body Pi (921600 baud) | Safe |
| Pad 29 | GPIO19 | `USB_DM` | Pogo Pad TP_DM | Native USB D- for JTAG / Firmware Flashing | Safe |
| Pad 30 | GPIO20 | `USB_DP` | Pogo Pad TP_DP | Native USB D+ for JTAG / Firmware Flashing | Safe |
| Pad 40 | **GPIO45**| `NC` | **Unconnected / Reserved** | **VDD_SPI Flash Voltage Strap — Left completely free!** | **ISOLATED** |
| Pad 41 | **GPIO46**| `NC` | **Unconnected / Reserved** | **ROM Boot Print Strap — Left completely free!** | **ISOLATED** |
| Pad 42 | GPIO47 | `TFT_RESET`| J_TFT Pin 10 (RESET) | Display Hardware Reset | Safe |
| Pad 43 | GPIO0 | `BOOT` | Pogo Pad TP_IO0 & R_BOOT (10k Pullup) | Boot Mode Strapping Pin (LOW = Download) | Strap (Handled) |

---

## 5. Camera & Display FPC Connector Pin Mapping

### 5.1 OV2640 24-Pin FPC Connector (`J_CAM`)
| Pin | Signal | Connected To | Function |
| :---: | :--- | :--- | :--- |
| 1, 2 | `NC / STROBE` | No Connection | Not used |
| 3 | `CAM_D7` | ESP32-S3 GPIO16 | Camera Data Bit 7 |
| 4 | `AGND` | System Ground | Analog Ground |
| 5 | `CAM_D6` | ESP32-S3 GPIO17 | Camera Data Bit 6 |
| 6 | `RESET` | `+3V3_SYS` via 10k + 100nF to GND | Hardware Reset with Power-On Delay |
| 7 | `CAM_D5` | ESP32-S3 GPIO18 | Camera Data Bit 5 |
| 8 | `CAM_SIOC` | ESP32-S3 GPIO40 | SCCB Clock (I2C) |
| 9 | `CAM_D4` | ESP32-S3 GPIO12 | Camera Data Bit 4 |
| 10 | `CAM_SIOD` | ESP32-S3 GPIO39 | SCCB Data (I2C) |
| 11 | `CAM_D3` | ESP32-S3 GPIO10 | Camera Data Bit 3 |
| 12 | `CAM_XCLK` | ESP32-S3 GPIO15 | Master Clock to Sensor |
| 13 | `CAM_D2` | ESP32-S3 GPIO8 | Camera Data Bit 2 |
| 14 | `CAM_VSYNC` | ESP32-S3 GPIO6 | Vertical Sync |
| 15 | `CAM_D1` | ESP32-S3 GPIO9 | Camera Data Bit 1 |
| 16 | `CAM_HREF` | ESP32-S3 GPIO7 | Horizontal Reference |
| 17 | `CAM_D0` | ESP32-S3 GPIO11 | Camera Data Bit 0 |
| 18 | `CAM_PCLK` | ESP32-S3 GPIO13 | Pixel Clock |
| 19 | `DGND` | System Ground | Digital Ground |
| 20 | `AVDD` | `+3V3_SYS` (via 10µF) | Sensor Analog Power (3.3V) |
| 21 | `DOVDD` | `+3V3_SYS` | Sensor I/O Power (3.3V) |
| 22 | `DVDD` | `+3V3_SYS` | Internal Core Regulator Supply |
| 23 | `PWDN` | GND via 10k pulldown | Power Down (Pulled LOW = Active) |
| 24 | `GND` | System Ground | Ground Return |

### 5.2 2.4" ILI9341 20-Pin FPC Connector (`J_TFT`)
| Pin | Signal | Connected To | Function |
| :---: | :--- | :--- | :--- |
| 1 | `GND` | System Ground | Ground Return |
| 2 | `VCC` | `+3V3_SYS` | Logic Power (3.3V) |
| 3, 4 | `NC` | No Connection | Internal panel test lines |
| 5 | `TFT_MOSI` | ESP32-S3 GPIO35 | SPI Master Out Slave In |
| 6 | `NC` | No Connection | MISO not required (Display Write Only) |
| 7 | `TFT_DC` | ESP32-S3 GPIO37 | Data / Command Selection (Non-strapping GPIO) |
| 8 | `TFT_SCK` | ESP32-S3 GPIO36 | SPI Serial Clock |
| 9 | `TFT_CS` | ESP32-S3 GPIO38 | SPI Chip Select (Non-strapping GPIO) |
| 10 | `TFT_RESET` | ESP32-S3 GPIO47 | Hardware Reset (Active Low) |
| 11 | `NC` | No Connection | Not used |
| 12, 13| `VCC_BL` | `+3V3_SYS` (via R_BL 10Ω) | Backlight LED Supply (Fixed 100% Brightness) |
| 14–19 | `NC` | No Connection | Reserved / Panel manufacturer lines |
| 20 | `GND` | System Ground | Ground Return |

---

## 6. Layout & Routing Guidelines for Head PCB

1. **Complete Isolation of Boot Strapping Pins:**
   - Verify in PCB layout DRC that `GPIO45`, `GPIO46`, and `GPIO3` pads have **no tracks or copper pours connecting them to anything**. Leaving them floating/open allows the ESP32-S3's internal weak pull-up/pull-down resistors to govern default boot behavior flawlessly.
2. **Servo Noise & Flyback Suppression:**
   - Place `D_SERVO` (SS14 Schottky diode) and `C_SERVO` (220µF radial capacitor) **immediately beside `J_SERVO`**.
   - Diode cathode to `+5V_SERVO` and anode to `GND`. This completely snubs reverse-voltage spikes induced by motor deceleration.
3. **Camera DVP Bus Length Matching:**
   - Route `CAM_D0`–`CAM_D7` and `CAM_PCLK` on Layer 1 over an unbroken Ground Plane, matched to within ±2.5mm (100 mil).
4. **Shielded Neck Cable Grounding:**
   - The outer braided shield of the 4-core neck cable must be soldered to **Body GND only** (left open at the Head end) to avoid ground loops while shielding UART TX/RX against servo noise.
