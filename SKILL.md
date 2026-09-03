# Architect AI — Model Training Plan

## 1. מטרת הפרויקט

לבנות מודל AI שמתפקד כ-"Architectural Planner".

המודל יקבל:
- דרישות משתמש
- מאפייני הפרויקט
- אילוצים תכנוניים
- אילוצים רגולטוריים שמגיעים בזמן Runtime

ויחזיר:
- Architectural Program
- חלוקת חללים
- גדלים מומלצים
- קשרים בין חללים
- Zoning
- Adjacency Graph
- Architectural SPEC מובנה

ה-SPEC יעבור לאחר מכן למנוע גיאומטרי שייצר
Floor Plan / SVG / DXF / CAD.

המטרה של ה-Fine-Tuning אינה ללמד את המודל
את חוקי מדינת ישראל.

המטרה היא ללמד אותו:

"HOW TO DESIGN"

בעוד שמנוע הרגולציה מספק:

"WHAT IS ALLOWED / REQUIRED"


---

# 2. ארכיטקטורת המערכת

User Requirements
        |
        v
+----------------------+
| Brief Parser         |
+----------------------+
        |
        v
Structured Brief
        |
        +-------------------------+
        |                         |
        v                         v
Regulation Engine           Site Information
(RAG / Rules DB)            (future)
        |                         |
        +------------+------------+
                     |
                     v
          Brief + Constraints
                     |
                     v
        +------------------------+
        | Architect LLM          |
        | Fine-Tuned Model       |
        +------------------------+
                     |
                     v
          Architectural SPEC
                     |
                     v
        +------------------------+
        | Geometry Solver        |
        | CP-SAT / Optimization  |
        +------------------------+
                     |
                     v
              Floor Plan
                     |
                     v
        +------------------------+
        | Validator              |
        +------------------------+
                     |
             +-------+-------+
             |               |
           VALID           INVALID
             |               |
             v               v
         Renderer       Revision Loop
             |
             v
        SVG / DXF / CAD


---

# 3. חלוקת האחריות

## Architect LLM

המודל אחראי על שיקול הדעת התכנוני.

לדוגמה:

- אילו חללים נדרשים
- חלוקה לאזור ציבורי / פרטי / שירות
- איזה חדר צריך להיות ליד איזה חדר
- הפרדת חדרי שינה מהאזור הציבורי
- קשר מטבח–פינת אוכל–סלון
- מיקום הגיוני של חדרי רחצה
- יצירת circulation הגיוני
- חלוקת שטחים
- יצירת מספר חלופות אפשריות


## Regulation Engine

אחראי על ידע שעשוי להשתנות.

לדוגמה:

- תקנות ישראליות
- דרישות ממ"ד
- נגישות
- בטיחות אש
- דרישות לפי סוג מבנה
- תקנות תכנון ובנייה
- אילוצים מקומיים
- הוראות תוכנית רלוונטית

המידע הזה אינו נצרב במודל.

הוא מוזרם בזמן Runtime.


## Geometry Solver

אחראי על המתמטיקה.

לדוגמה:

- קואורדינטות
- polygons
- מניעת overlap
- עובי קירות
- חיבור חדרים
- מיקום פתחים
- התאמה לגבולות המבנה
- אופטימיזציה של שטחים


## Validator

אחראי לוודא שהתוצאה באמת עומדת בדרישות.

ה-LLM אינו מקור האמת לגבי Compliance.


---

# 4. Dataset בסיסי

נקודת הפתיחה:

BOOMI Stage-A Text → SPEC

המבוסס על ResPlan.

BOOMI מספק דוגמאות מהצורה:

Natural Language
        ↓
Architectural SPEC

ה-SPEC כולל בין היתר:

- plot
- BHK
- room program
- room types
- target areas
- adjacency relationships


---

# 5. Dataset שאנחנו ניצור

לא נאמן ישירות על BOOMI.

נייצר Dataset חדש:

Architect Dataset V1


## Training Input

{
    "brief": {...},

    "constraints": [...]
}


## Training Output

{
    "program": {...},

    "zones": [...],

    "rooms": [...],

    "relationships": [...],

    "circulation": [...],

    "architectural_spec": {...}
}


---

# 6. Brief

ה-Brief מייצג את דרישות המשתמש.

דוגמה:

{
    "building_type": "residential",
    "built_area_m2": 180,

    "bedrooms": 4,
    "bathrooms": 2,

    "floors": 1,

    "preferences": {
        "open_kitchen": true,
        "parents_suite": true,
        "large_living_room": true
    }
}

חשוב:

Brief הוא רצון המשתמש.

הוא אינו חוק.


---

# 7. Constraints

Constraints הם תנאים שהתכנון חייב לנסות לקיים.

לדוגמה:

{
    "type": "required_room",
    "room": "SAFE_ROOM"
}

או:

{
    "type": "min_area",
    "room": "BEDROOM",
    "value": 9
}

או:

{
    "type": "required_adjacency",
    "from": "KITCHEN",
    "to": "DINING"
}


---

# 8. Hard Constraints מול Soft Constraints

חשוב להבדיל ביניהם.


## Hard Constraint

אסור להפר.

לדוגמה:

- דרישה רגולטורית
- חדר חובה
- רוחב מינימלי
- גבול מגרש
- איסור overlap


## Soft Constraint

העדפה תכנונית.

לדוגמה:

- רצוי מטבח ליד פינת אוכל
- רצוי פרטיות לחדרי שינה
- המשתמש מעדיף סלון גדול
- עדיפות למעט מסדרונות


ה-Solver יכול לבצע Optimization על Soft Constraints.


---

# 9. יצירת Training Samples

מתוכנית אחת לא ניצור בהכרח Sample אחד.

נוכל ליצור מספר וריאציות.


Original Plan
      |
      +--> Sample A
      |
      +--> Sample B
      |
      +--> Sample C
      |
      +--> Sample D


לדוגמה:


### Sample A

Brief:

4 bedrooms
180 sqm

Constraints:

none / minimal


### Sample B

Brief:

4 bedrooms
180 sqm

Constraints:

Kitchen adjacent to living


### Sample C

Brief:

4 bedrooms
180 sqm

Constraints:

Balcony required


### Sample D

Brief:

4 bedrooms
180 sqm

Constraints:

Balcony required
Kitchen adjacent to living


Output:

אותו Architectural SPEC יכול להיות Target
אם הוא מקיים את כל ה-Constraints.


---

# 10. למה לעשות Constraint Augmentation?

אנחנו לא רוצים שהמודל ילמד:

INPUT X
=
PLAN Y


אנחנו רוצים שילמד:

Brief
+
Constraints
+
Architectural Principles

        ↓

One valid architectural solution


כלומר קיימים פתרונות רבים לאותה בעיה.


---

# 11. Constraint Generator

נבנה רכיב:

constraint_generator.py


הוא יקבל תוכנית קיימת ויזהה עובדות שהיא מקיימת.

לדוגמה:

Plan:

Kitchen ↔ Living

Bedroom ↔ Bathroom

Living ↔ Balcony


אפשר ליצור ממנו:

Constraint Set A:

Kitchen adjacent Living


Constraint Set B:

Living connected Balcony


Constraint Set C:

Kitchen adjacent Living
+
Living connected Balcony


לעולם לא ניצור Constraint שה-Solution אינו מקיים.


---

# 12. Negative Examples

בשלב מתקדם ניצור גם תוכניות בעייתיות.

לדוגמה:

- חדר ללא גישה
- overlap בין חדרים
- circulation גרוע
- חדר רחצה במקום לא הגיוני
- שטח בלתי אפשרי
- adjacency לא הגיוני


וניצור Dataset נוסף:

Architect Critic Dataset


INPUT:

Brief
+
Constraints
+
Plan


OUTPUT:

{
    "valid": false,

    "problems": [...],

    "suggested_changes": [...]
}


---

# 13. Architect Model מול Critic

אפשר להתחיל עם אותו מודל.

בהמשך ניתן להפריד:


Architect Model
       |
       v
Proposed SPEC
       |
       v
Critic
       |
       +---- GOOD ----> Solver
       |
       |
       +---- BAD
              |
              v
         Revision
              |
              v
        Architect Model


---

# 14. Fine-Tuning Strategy

שלב ראשון:

Supervised Fine-Tuning באמצעות LoRA / QLoRA.


Base Model
     |
     v
Architect Dataset
     |
     v
LoRA Fine-Tuning
     |
     v
Architect Model V1


לא משנים את כל משקלי המודל.

נשתמש ב-PEFT / LoRA.


---

# 15. מה המודל צריך ללמוד

ה-Fine-Tuning צריך ללמד דפוסים כגון:

### Space Programming

איך לחלק שטח נתון בין חדרים.


### Adjacency

איזה חללים צריכים להיות קשורים.


### Zoning

Public

Private

Service


### Circulation

איך אנשים עוברים בבית.


### Privacy

הפרדה בין אזורים ציבוריים ופרטיים.


### Functional Relationships

לדוגמה:

Kitchen
   ↓
Dining
   ↓
Living


### Constraint Following

איך לשנות תכנון כאשר מגיע Constraint חדש.


---

# 16. מה המודל לא צריך ללמוד

לא ננסה לצרוב ב-Fine-Tuning:

- חוקי מדינת ישראל
- תקנות משתנות
- הוראות רשות מקומית
- מספרי סעיפים
- תוכניות בניין עיר
- דרישות שמתעדכנות

אלו יהיו Runtime Knowledge.


---

# 17. Regulation Dataset

נבנה Dataset / Database נפרד.

לדוגמה:

{
    "id": "REG-XXXX",

    "jurisdiction": "Israel",

    "building_type": "residential",

    "condition": {...},

    "constraint": {...},

    "source": {...},

    "effective_date": "...",

    "status": "active"
}


Regulation Engine:

Project
   ↓
Find applicable regulations
   ↓
Convert regulations
to machine constraints
   ↓
Architect Model


---

# 18. Production Prompt

בזמן אמת המודל יקבל בערך:


SYSTEM

You are an architectural planning model.

Generate a functional architectural solution
that satisfies all HARD constraints.

Optimize SOFT constraints when possible.


USER

Brief:
{...}

Hard Constraints:
[...]

Soft Constraints:
[...]

Site Constraints:
[...]


OUTPUT:

Architectural SPEC JSON


---

# 19. Geometry

המודל לא חייב להחליט ישירות:

x = 3.284
y = 7.932


במקום זאת הוא מייצר:

Room Program
+
Areas
+
Relationships
+
Zones
+
Constraints


ואז:

CP-SAT / Geometry Solver

מחשב:

- x
- y
- width
- height
- polygon
- walls
- doors
- windows


---

# 20. Validation Loop

לא סומכים על יציאה אחת של המודל.


Architect
   ↓
SPEC
   ↓
Solver
   ↓
Validator
   ↓

PASS?
   |
 +---- YES ---> Render
 |
 +---- NO
        |
        v
 Violations
        |
        v
 Architect
        |
        v
 Revised SPEC


מגבילים את מספר ניסיונות התיקון.


---

# 21. Evaluation

לפני Production צריך Golden Test Set.

נמדוד לפחות:


## JSON Validity

האם הפלט עומד ב-Schema.


## Constraint Satisfaction Rate

כמה מה-Hard Constraints התקיימו.


## Program Accuracy

האם כל החדרים שהתבקשו קיימים.


## Area Error

הפער בין השטח המבוקש למתקבל.


## Adjacency Accuracy

האם הקשרים המבוקשים מתקיימים.


## Solver Success Rate

באיזה אחוז מה-SPECs ניתן בכלל לייצר Geometry.


## Regulatory Compliance

כמה מהבדיקות הדטרמיניסטיות עברו.


## Architectural Quality

Critic / Human Architect evaluation.


---

# 22. Dataset Split

חשוב מאוד:

אסור שווריאציות של אותה תוכנית יהיו גם ב-Train וגם ב-Test.

נבצע Split לפי:

source_plan_id


לדוגמה:

80% plans → Train

10% plans → Validation

10% plans → Test


ורק לאחר מכן נייצר Variations.


---

# 23. שלבי העבודה

## Phase 1 — Dataset Exploration

- הורדת BOOMI
- בדיקת schema
- ניתוח room types
- ניתוח adjacency
- בדיקת איכות
- בדיקת חריגים


## Phase 2 — Normalization

יצירת vocabulary אחיד:

BEDROOM
LIVING
KITCHEN
DINING
BATHROOM
WC
BALCONY
CORRIDOR
...


## Phase 3 — Architect Dataset V1

BOOMI
↓
Preprocessor
↓
Brief
↓
Constraints
↓
Architectural SPEC


## Phase 4 — Dataset QA

מתחילים רק עם:

100 samples

בודקים ידנית.

לאחר מכן:

1,000 samples

ורק לאחר שהכול תקין:

Full Dataset


## Phase 5 — Baseline

לפני Fine-Tuning:

Base Model
+
Few-shot examples

נמדוד ביצועים.


## Phase 6 — Fine-Tuning V1

LoRA / QLoRA

Brief + Constraints
        ↓
Architectural SPEC


## Phase 7 — Evaluation

Base Model

VS

Fine-Tuned Architect


## Phase 8 — Geometry Solver

Architectural SPEC
        ↓
CP-SAT
        ↓
Geometry


## Phase 9 — Renderer

Geometry
   ↓

SVG

ולאחר מכן:

DXF / CAD


## Phase 10 — Israeli Regulation Engine

מקורות רשמיים
      ↓
Extraction
      ↓
Normalized Rules
      ↓
Constraints
      ↓
Runtime


## Phase 11 — Validator

בדיקות:

Geometry
+
Brief
+
Regulations


## Phase 12 — Architect Critic

Dataset של:

Good Plans

vs

Bad Plans

ללמד את המערכת לזהות תכנון בעייתי.


---

# 24. יעד V1

היעד הראשון אינו:

"להחליף אדריכל בישראל."


היעד הוא:

User Brief
     ↓
Architect Model
     ↓
Architectural SPEC
     ↓
Solver
     ↓
Valid Floor Plan


עבור:

Residential
Single Floor
Simple Geometry


ורק אחרי שזה עובד היטב:

Multi-floor
↓
MAMAD
↓
Site constraints
↓
Accessibility
↓
Commercial
↓
Schools
↓
Mixed-use
↓
Full Israeli regulatory system


---

# 25. עיקרון מרכזי

המערכת בנויה מארבעה סוגים שונים של Intelligence:


1. Learned Architectural Intelligence

Fine-Tuned Model

"איך מתכננים?"


2. Current Regulatory Knowledge

RAG / Regulation DB

"מה נדרש כרגע?"


3. Mathematical Intelligence

Solver

"האם אפשר לגרום לזה להסתדר?"


4. Verification

Validator

"האם התוצאה באמת תקינה?"


הפרדת ארבעת התפקידים האלה היא הבסיס של המערכת.