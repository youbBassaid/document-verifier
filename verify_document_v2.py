# file: verify_document_v2.py
# Advanced Certificate Verification System with Higher Precision

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
import cv2
import numpy as np
from rapidfuzz import fuzz
import re
import io
from typing import Dict, List, Tuple, Optional
import unicodedata

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
# Enable CORS for frontend at localhost:3000
CORS(app, origins=["http://localhost:3000"], supports_credentials=True)

class AdvancedImagePreprocessor:
    """Multiple preprocessing strategies for better OCR accuracy"""
    
    @staticmethod
    def preprocess_standard(img):
        """Standard preprocessing"""
        # Resize if too small
        height, width = img.shape[:2]
        if height < 1500:
            scale = 2000 / height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    
    @staticmethod
    def preprocess_denoise(img):
        """Heavy denoising for low quality images"""
        height, width = img.shape[:2]
        if height < 1500:
            scale = 2000 / height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Stronger denoising
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    
    @staticmethod
    def preprocess_morphology(img):
        """Morphological operations for better text clarity"""
        height, width = img.shape[:2]
        if height < 1500:
            scale = 2000 / height
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Morphological gradient
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        _, th = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th

class MultiStrategyOCR:
    """Multiple OCR strategies for better extraction"""
    
    @staticmethod
    def extract_with_multiple_configs(img_array) -> List[str]:
        """Try multiple Tesseract configurations and return all results"""
        pil = Image.fromarray(img_array)
        results = []
        
        # Configuration 1: Default with all languages
        configs = [
            r'--oem 3 --psm 6',
            r'--oem 3 --psm 3',
            r'--oem 3 --psm 4',
            r'--oem 1 --psm 6',
        ]
        
        for config in configs:
            try:
                text = pytesseract.image_to_string(pil, lang='ara+fra+eng', config=config)
                if len(text.strip()) > 50:
                    results.append(text)
            except:
                continue
        
        return results

class SmartExtractor:
    """Intelligent extraction with confidence scoring"""
    
    @staticmethod
    def extract_student_name(texts: List[str]) -> Tuple[str, float]:
        """Extract student name with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(?:الطالب|Student|Étudiant)\s*(?:\(ة\))?\s*([A-Z][A-Za-z]+(?:\s+[A-Za-z]+)+)', 0.95),
                (r'\b([A-Z][A-Z]{2,})\s+([a-z]{3,})\b', 0.85),
                (r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', 0.75),
                (r'(?:السيـــد|السيد)\s*(?:\(ة\))?\s*[:：]\s*([^\n]{5,50})', 0.90),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text, re.MULTILINE)
                for match in matches:
                    if match.lastindex == 2:
                        name = f"{match.group(1)} {match.group(2)}"
                    else:
                        name = match.group(1)
                    
                    name = name.strip()
                    
                    # Filter false positives
                    excluded = ['Republique', 'Algerionne', 'Democratique', 'Populaire', 
                              'Ministere', 'Enseignement', 'Superieur', 'Recherche',
                              'Scientifique', 'Universite', 'Master', 'License',
                              'Mathematiques', 'Informatique']
                    
                    if (len(name) > 5 and len(name) < 50 and
                        not any(word.lower() in name.lower() for word in excluded) and
                        not name.startswith(('UN', 'http')) and
                        not re.match(r'^\d+', name)):
                        candidates.append((name, confidence))
        
        # Return most confident result
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)
    
    @staticmethod
    def extract_birth_date(texts: List[str]) -> Tuple[str, float]:
        """Extract birth date with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(?:المولود|في|né|born|date)\s*(?:\(ة\))?\s*(?:في)?\s*[:：]\s*(\d{4}[/\-\.\s]\d{1,2}[/\-\.\s]\d{1,2})', 0.95),
                (r'(?:المولود|في|né|born|date)\s*(?:\(ة\))?\s*(?:في)?\s*[:：]\s*(\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{4})', 0.95),
                (r'\b(\d{4}[/\-\.\s]\d{1,2}[/\-\.\s]\d{1,2})\b', 0.70),
                (r'\b(\d{1,2}[/\-\.\s]\d{1,2}[/\-\.\s]\d{4})\b', 0.70),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    date_str = match.group(1).strip()
                    date_str = date_str.replace(' ', '/').replace('.', '/').replace('-', '/')
                    
                    parts = date_str.split('/')
                    if len(parts) == 3:
                        try:
                            if len(parts[0]) == 4:
                                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                            else:
                                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
                            
                            if 1950 <= year <= 2010 and 1 <= month <= 12 and 1 <= day <= 31:
                                candidates.append((date_str, confidence))
                        except:
                            continue
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)
    
    @staticmethod
    def extract_registration_number(texts: List[str]) -> Tuple[str, float]:
        """Extract registration number with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(?:رقم\s*التسجيل|رقم\s*التسجيـــل|Registration|Matricule)\s*[:：]\s*(UN\s*\d[\d\s]{10,})', 0.95),
                (r'(?:رقم\s*التسجيل|رقم\s*التسجيـــل|Registration|Matricule)\s*[:：]\s*(\d{10,})', 0.95),
                (r'\b(UN\s*\d[\d\s]{10,})\b', 0.80),
                (r'(?:رقم|رقـــم)\s*[:：]\s*(\d{10,})', 0.75),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    reg_num = match.group(1).replace(' ', '').replace('\u200f', '').replace('\u200e', '')
                    
                    if reg_num.replace('UN', '').isdigit() and len(reg_num.replace('UN', '')) >= 10:
                        candidates.append((reg_num, confidence))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)
    
    @staticmethod
    def extract_degree_level(texts: List[str]) -> Tuple[str, float]:
        """Extract degree level with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(?:السنة|Level|Niveau)\s*[:：]\s*([^\n]{3,100})', 0.95),
                (r'(الأولى\s*-\s*الماستر)', 0.90),
                (r'(ماستر\s+سنة\s+\w+)', 0.90),
                (r'(ليسانس\s+سنة\s+\w+)', 0.90),
                (r'(Master\s+\d+)', 0.85),
                (r'(License\s+\d+)', 0.85),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    degree = match.group(1).strip()
                    degree = re.split(r'[،,\n/]', degree)[0].strip()
                    
                    if len(degree) > 3:
                        # Normalize
                        if 'ماستر' in degree or 'الماستر' in degree or 'master' in degree.lower():
                            if any(x in degree for x in ['ثانية', 'الثانية', '2', 'deux']):
                                degree = 'Master 2'
                            elif any(x in degree for x in ['أولى', 'الأولى', '1', 'première', 'أول']):
                                degree = 'Master 1'
                            else:
                                degree = 'Master'
                        elif 'ليسانس' in degree or 'license' in degree.lower():
                            if any(x in degree for x in ['ثالثة', 'الثالثة', '3', 'troisième']):
                                degree = 'License 3'
                            elif any(x in degree for x in ['ثانية', 'الثانية', '2', 'deuxième']):
                                degree = 'License 2'
                            elif any(x in degree for x in ['أولى', 'الأولى', '1', 'première', 'أول']):
                                degree = 'License 1'
                        
                        candidates.append((degree, confidence))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)
    
    @staticmethod
    def extract_academic_year(texts: List[str]) -> Tuple[str, float]:
        """Extract academic year with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(?:السنة\s+الجامعية|السنة\s*الجامعـــية|خلال)\s*[:：]\s*(\d{4}[/\-:]\d{4})', 0.95),
                (r'\b(\d{4}[/\-:]\d{4})\b', 0.75),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    year_str = match.group(1).replace(':', '/').replace('-', '/')
                    parts = year_str.split('/')
                    
                    if len(parts) == 2:
                        try:
                            year1, year2 = int(parts[0]), int(parts[1])
                            if abs(year1 - year2) == 1 and 2000 <= min(year1, year2) <= 2050:
                                candidates.append((year_str, confidence))
                        except:
                            continue
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)
    
    @staticmethod
    def extract_university(texts: List[str]) -> Tuple[str, float]:
        """Extract university with confidence score"""
        candidates = []
        
        for text in texts:
            patterns = [
                (r'(جامعة\s+[\u0600-\u06FF\s]{3,40})', 0.95),
                (r'(Université\s+(?:de\s+)?(?:la\s+)?[A-Za-z\s]{3,40})', 0.90),
                (r'université\s+(?:de\s+)?([A-Za-z\s]{3,30})', 0.85),
            ]
            
            for pattern, confidence in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    uni_name = match.group(1).strip() if match.lastindex == 1 else match.group(0).strip()
                    uni_name = re.split(r'\n|\r', uni_name)[0].strip()
                    uni_name = ' '.join(uni_name.split())
                    
                    if 5 < len(uni_name) < 100:
                        uni_name = re.sub(r'\s*\d+\s*$', '', uni_name).strip()
                        candidates.append((uni_name, confidence))
        
        # Fallback: check for specific universities
        if not candidates:
            text_combined = ' '.join(texts).lower()
            universities = {
                ('ghardaia', 'gharders', 'غرداية', 'عردايه'): ('Université de Ghardaïa', 0.85),
                ('الجزائر 3', 'algerie 3', 'alger 3'): ('Université d\'Alger 3', 0.85),
                ('تيسمسيلت', 'tissemsilt'): ('Université de Tissemsilt', 0.85),
                ('التكوين المتواصل', 'formation continue'): ('Université de la Formation Continue', 0.85),
            }
            
            for keywords, (uni_name, conf) in universities.items():
                if any(kw in text_combined or kw in ' '.join(texts) for kw in keywords):
                    candidates.append((uni_name, conf))
                    break
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        return ('', 0.0)

def detect_document_authenticity(img) -> Tuple[bool, str, Dict]:
    """Detect if document appears to be an authentic certificate based on visual features"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Feature 1: Edge density (official certificates have logos, borders, seals)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / edges.size
    
    # Feature 2: Contour complexity (official docs have multiple design elements)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Feature 3: Color variance (official certificates use multiple colors, not just black on white)
    color_std = np.std(img)
    
    # Feature 4: Text layout complexity (official docs have structured layouts)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text_regions = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    text_regions = [r for r in text_regions if cv2.contourArea(r) > 50]
    
    features = {
        'edge_density': round(edge_density, 4),
        'contour_count': len(significant_contours),
        'color_variance': round(float(color_std), 2),
        'text_regions': len(text_regions)
    }
    
    # Simple documents have:
    # - Low edge density (no logos/seals)
    # - Few contours (simple text layout)
    # - Low color variance (mostly white background with black text)
    # - Few text regions (simple centered text)
    
    is_simple_document = (
        edge_density < 0.05 and  # Very few edges (no complex graphics)
        len(significant_contours) < 50 and  # Few design elements
        color_std < 30  # Very uniform (plain white background)
    )
    
    if is_simple_document:
        return False, "Document appears to be a simple typed document, not an official certificate. Official certificates contain university logos, official seals, borders, and complex formatting.", features
    
    # Also check for suspiciously perfect/minimal layout
    if len(text_regions) < 10:
        return False, "Document layout is too simple. Official certificates have complex structured layouts with multiple sections, headers, and official elements.", features
    
    return True, "Document appears authentic", features

def extract_certificate_info_advanced(img_bytes) -> Dict:
    """Advanced extraction with multiple strategies"""
    
    # Decode image
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    
    # FIRST: Check document authenticity based on visual features
    is_authentic, auth_reason, visual_features = detect_document_authenticity(img)
    if not is_authentic:
        return {
            'is_authentic': False,
            'authenticity_reason': auth_reason,
            'visual_features': visual_features,
            'extracted_data': {},
            'confidence_scores': {},
            'raw_texts': []
        }
    
    # Try multiple preprocessing methods
    preprocessor = AdvancedImagePreprocessor()
    preprocessed_images = [
        preprocessor.preprocess_standard(img.copy()),
        preprocessor.preprocess_denoise(img.copy()),
        preprocessor.preprocess_morphology(img.copy()),
    ]
    
    # Extract text with multiple strategies
    ocr = MultiStrategyOCR()
    all_texts = []
    for prep_img in preprocessed_images:
        texts = ocr.extract_with_multiple_configs(prep_img)
        all_texts.extend(texts)
    
    # Smart extraction with confidence
    extractor = SmartExtractor()
    student_name, name_conf = extractor.extract_student_name(all_texts)
    birth_date, dob_conf = extractor.extract_birth_date(all_texts)
    registration_number, reg_conf = extractor.extract_registration_number(all_texts)
    degree_level, degree_conf = extractor.extract_degree_level(all_texts)
    academic_year, year_conf = extractor.extract_academic_year(all_texts)
    university, uni_conf = extractor.extract_university(all_texts)
    
    return {
        'extracted_data': {
            'student_name': student_name,
            'birth_date': birth_date,
            'registration_number': registration_number,
            'degree_level': degree_level,
            'academic_year': academic_year,
            'university': university,
        },
        'confidence_scores': {
            'student_name': round(name_conf, 3),
            'birth_date': round(dob_conf, 3),
            'registration_number': round(reg_conf, 3),
            'degree_level': round(degree_conf, 3),
            'academic_year': round(year_conf, 3),
            'university': round(uni_conf, 3),
        },
        'raw_texts': [text[:500] for text in all_texts[:2]],  # First 2 for debugging
    }

def compare_strings(a, b):
    """Compare two strings using fuzzy matching - case insensitive"""
    if not a or not b:
        return 0.0
    # Convert both to lowercase for case-insensitive comparison
    return fuzz.token_set_ratio(a.lower(), b.lower()) / 100.0

def normalize_date(date_str):
    """Normalize date format"""
    if not date_str:
        return []
    date_str = date_str.replace(' ', '').replace('-', '/').replace('.', '/')
    parts = date_str.split('/')
    if len(parts) == 3:
        if len(parts[0]) == 4:
            return [date_str, f"{parts[2]}/{parts[1]}/{parts[0]}"]
        else:
            return [date_str, f"{parts[2]}/{parts[1]}/{parts[0]}"]
    return [date_str]

def normalize_degree(degree_str):
    """Normalize degree format to standardized version.
    
    Handles variations like:
    - "2_master" -> "Master 2"
    - "Master2" -> "Master 2"
    - "master 2" -> "Master 2"
    - "ماستر سنة ثانية" -> "Master 2"
    - "License 3" -> "License 3"
    """
    if not degree_str:
        return ''
    
    # Clean string: remove non-alphanumeric and Arabic chars, normalize spacing
    degree_lower = degree_str.lower().strip()
    degree_lower = re.sub(r'[^a-z0-9\u0600-\u06FF\s]', ' ', degree_lower)
    degree_lower = ' '.join(degree_lower.split())  # collapse multiple spaces

    # Normalize Arabic degree levels
    if 'ماستر' in degree_lower or 'الماستر' in degree_lower or 'master' in degree_lower:
        if any(x in degree_lower for x in ['ثانية', 'الثانية', 'دوم', 'second', 'deux', '2']):
            return 'Master 2'
        elif any(x in degree_lower for x in ['أولى', 'الأولى', 'أول', 'first', 'première', '1']):
            return 'Master 1'
        else:
            return 'Master'
    
    if 'ليسانس' in degree_lower or 'license' in degree_lower or 'licence' in degree_lower:
        if any(x in degree_lower for x in ['ثالثة', 'الثالثة', 'third', 'troisième', '3']):
            return 'License 3'
        elif any(x in degree_lower for x in ['ثانية', 'الثانية', 'second', 'deuxième', '2']):
            return 'License 2'
        elif any(x in degree_lower for x in ['أولى', 'الأولى', 'أول', 'first', 'première', '1']):
            return 'License 1'
        else:
            return 'License'
    
    # Normalize Master variations
    if 'master' in degree_lower:
        if any(x in degree_lower for x in ['2', 'two', 'deux', 'second', 'ثانية']):
            return 'Master 2'
        elif any(x in degree_lower for x in ['1', 'one', 'une', 'first', 'première']):
            return 'Master 1'
        else:
            return 'Master'
    
    # Normalize License variations
    if 'license' in degree_lower or 'licence' in degree_lower:
        if any(x in degree_lower for x in ['3', 'three', 'trois', 'third']):
            return 'License 3'
        elif any(x in degree_lower for x in ['2', 'two', 'deux', 'second']):
            return 'License 2'
        elif any(x in degree_lower for x in ['1', 'one', 'une', 'first']):
            return 'License 1'
        else:
            return 'License'
    
    # Return original if no match
    return degree_str

@app.route('/verify', methods=['POST'])
def verify():
    """Advanced verification endpoint"""
    user_name = request.form.get('user_name', '').strip()
    user_dob = request.form.get('user_dob', '').strip()
    user_degree_level = request.form.get('user_degree_level', '').strip()
    user_academic_year = request.form.get('user_academic_year', '').strip()
    file = request.files.get('doc')
    
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Advanced extraction
    img_bytes = file.read()
    result = extract_certificate_info_advanced(img_bytes)
    
    # Check if document failed visual authenticity test
    if result.get('is_authentic') == False:
        return jsonify({
            'success': False,
            'error': 'Invalid document - Not an official certificate',
            'reason': result.get('authenticity_reason'),
            'visual_features': result.get('visual_features'),
            'extracted_data': result.get('extracted_data', {})
        }), 400
    
    cert_info = result['extracted_data']
    confidence = result['confidence_scores']
    
    # DOCUMENT AUTHENTICITY VALIDATION
    # Check 1: University MUST be detected with high confidence (official certificates have letterheads)
    if not cert_info.get('university') or confidence['university'] < 0.80:
        return jsonify({
            'success': False,
            'error': 'Invalid document - Not an official certificate',
            'reason': 'No official university letterhead detected. This appears to be a simple document, not an authentic certificate. Please upload an official certificate from a recognized university.',
            'university_found': cert_info.get('university', 'None'),
            'university_confidence': confidence['university']
        }), 400
    
    # Check 2: Registration number is MANDATORY for authentic certificates
    if not cert_info.get('registration_number') or confidence['registration_number'] < 0.70:
        return jsonify({
            'success': False,
            'error': 'Invalid document - No registration number',
            'reason': 'Official certificates must have a student registration number. This document appears to be a simple note or form, not an authentic certificate.',
            'registration_found': cert_info.get('registration_number', 'None'),
            'registration_confidence': confidence['registration_number']
        }), 400
    
    # Check 3: ALL critical fields must be present with good confidence
    critical_fields_missing = []
    if not cert_info.get('student_name') or confidence['student_name'] < 0.70:
        critical_fields_missing.append('student_name')
    if not cert_info.get('degree_level') or confidence['degree_level'] < 0.70:
        critical_fields_missing.append('degree_level')
    if not cert_info.get('academic_year') or confidence['academic_year'] < 0.70:
        critical_fields_missing.append('academic_year')
    if not cert_info.get('birth_date') or confidence['birth_date'] < 0.60:
        critical_fields_missing.append('birth_date')
    
    if len(critical_fields_missing) >= 2:
        return jsonify({
            'success': False,
            'error': 'Invalid document - Incomplete certificate',
            'reason': 'This document is missing essential certificate information. Official certificates contain complete student records. Please upload a complete, official certificate.',
            'missing_or_low_confidence_fields': critical_fields_missing
        }), 400
    
    # Check 4: Minimum average confidence across all fields
    avg_confidence = sum(confidence.values()) / len(confidence)
    if avg_confidence < 0.65:
        return jsonify({
            'success': False,
            'error': 'Invalid document - Poor quality',
            'reason': 'Document quality is too low or structure does not match official certificates. Please upload a clear, official certificate.',
            'average_confidence': round(avg_confidence, 3),
            'required_minimum': 0.65
        }), 400
    
    # Verification
    verification = {}
    
    # Name verification (fuzzy)
    if user_name:
        name_score = compare_strings(user_name, cert_info['student_name'])
        verification['name_match_score'] = round(name_score, 3)
        verification['name_match'] = name_score >= 0.75
        verification['name_confidence'] = confidence['student_name']
    
    # DOB verification (exact after normalization)
    if user_dob:
        extracted_dob_formats = normalize_date(cert_info['birth_date'])
        user_dob_formats = normalize_date(user_dob)
        
        dob_match = False
        for user_fmt in user_dob_formats:
            for extracted_fmt in extracted_dob_formats:
                if user_fmt == extracted_fmt:
                    dob_match = True
                    break
            if dob_match:
                break
        
        verification['dob_match'] = dob_match
        verification['dob_match_score'] = 1.0 if dob_match else 0.0
        verification['dob_confidence'] = confidence['birth_date']
    
    # Degree verification (case-insensitive after normalization)
    if user_degree_level:
        normalized_user_degree = normalize_degree(user_degree_level)
        normalized_cert_degree = normalize_degree(cert_info['degree_level'])
        # Case-insensitive comparison
        degree_match = (normalized_user_degree.lower() == normalized_cert_degree.lower())
        verification['degree_match'] = degree_match
        verification['degree_match_score'] = 1.0 if degree_match else 0.0
        verification['degree_confidence'] = confidence['degree_level']
    
    # Academic year verification (exact with reversal tolerance)
    if user_academic_year:
        extracted_year = cert_info['academic_year'].replace('-', '/').replace(' ', '')
        user_year = user_academic_year.replace('-', '/').replace(' ', '')
        
        year_match = extracted_year == user_year
        if not year_match:
            parts = user_year.split('/')
            if len(parts) == 2:
                reversed_year = f"{parts[1]}/{parts[0]}"
                year_match = (extracted_year == reversed_year)
        
        verification['academic_year_match'] = year_match
        verification['academic_year_match_score'] = 1.0 if year_match else 0.0
        verification['academic_year_confidence'] = confidence['academic_year']
    
    # Calculate overall score
    if any([user_name, user_dob, user_degree_level, user_academic_year]):
        # CRITICAL: If academic year is provided but doesn't match, reject immediately
        if user_academic_year and not verification.get('academic_year_match', False):
            verification['final_score'] = 0.0
            verification['overall_confidence'] = round(sum(confidence.values()) / len(confidence), 3)
            verification['decision'] = 'rejected'
            verification['rejection_reason'] = 'Academic year mismatch'
        else:
            weights, scores = [], []
            
            if user_name:
                weights.append(0.4)
                scores.append(verification.get('name_match_score', 0))
            if user_dob:
                weights.append(0.3)
                scores.append(verification.get('dob_match_score', 0))
            if user_degree_level:
                weights.append(0.15)
                scores.append(verification.get('degree_match_score', 0))
            if user_academic_year:
                weights.append(0.15)
                scores.append(verification.get('academic_year_match_score', 0))
            
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                final_score = sum(s * w for s, w in zip(scores, weights))
            else:
                final_score = 0.0
            
            # Average confidence
            avg_confidence = sum(confidence.values()) / len(confidence)
            
            verification['final_score'] = round(final_score, 3)
            verification['overall_confidence'] = round(avg_confidence, 3)
            verification['decision'] = (
                'accepted' if final_score >= 0.85 and avg_confidence >= 0.80
                else ('review' if final_score >= 0.6 or avg_confidence >= 0.70 else 'rejected')
            )
    
    return jsonify({
        'success': True,
        'extracted_data': cert_info,
        'confidence_scores': confidence,
        'verification': verification if verification else None,
        'user_provided': {
            'name': user_name or None,
            'dob': user_dob or None,
            'degree_level': user_degree_level or None,
            'academic_year': user_academic_year or None,
        },
        'raw_text_samples': result['raw_texts'],
    })

if __name__ == '__main__':
    # Allow connections from other devices on the network
    app.run(debug=True, host='0.0.0.0', port=5001)