/********************************************************************************
** Form generated from reading UI file 'mainUI.ui'
**
** Created by: Qt User Interface Compiler version 5.9.5
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINUI_H
#define UI_MAINUI_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_guiDlg
{
public:
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer_5;
    QPushButton *configuration;
    QSpacerItem *horizontalSpacer;
    QPushButton *regenerate;
    QSpacerItem *horizontalSpacer_4;
    QLabel *label_2;
    QLineEdit *contributor;
    QSpacerItem *horizontalSpacer_3;
    QPushButton *quit;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label;

    void setupUi(QWidget *guiDlg)
    {
        if (guiDlg->objectName().isEmpty())
            guiDlg->setObjectName(QStringLiteral("guiDlg"));
        guiDlg->resize(1042, 819);
        verticalLayout = new QVBoxLayout(guiDlg);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalSpacer_5 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_5);

        configuration = new QPushButton(guiDlg);
        configuration->setObjectName(QStringLiteral("configuration"));
        configuration->setCheckable(true);

        horizontalLayout->addWidget(configuration);

        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        regenerate = new QPushButton(guiDlg);
        regenerate->setObjectName(QStringLiteral("regenerate"));

        horizontalLayout->addWidget(regenerate);

        horizontalSpacer_4 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_4);

        label_2 = new QLabel(guiDlg);
        label_2->setObjectName(QStringLiteral("label_2"));

        horizontalLayout->addWidget(label_2);

        contributor = new QLineEdit(guiDlg);
        contributor->setObjectName(QStringLiteral("contributor"));

        horizontalLayout->addWidget(contributor);

        horizontalSpacer_3 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_3);

        quit = new QPushButton(guiDlg);
        quit->setObjectName(QStringLiteral("quit"));

        horizontalLayout->addWidget(quit);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        verticalLayout->addLayout(horizontalLayout);

        label = new QLabel(guiDlg);
        label->setObjectName(QStringLiteral("label"));
        QSizePolicy sizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
        sizePolicy.setHorizontalStretch(1);
        sizePolicy.setVerticalStretch(1);
        sizePolicy.setHeightForWidth(label->sizePolicy().hasHeightForWidth());
        label->setSizePolicy(sizePolicy);
        label->setMinimumSize(QSize(1024, 768));
        label->setMaximumSize(QSize(10240, 7680));
        label->setSizeIncrement(QSize(1, 1));
        label->setScaledContents(true);

        verticalLayout->addWidget(label);


        retranslateUi(guiDlg);

        QMetaObject::connectSlotsByName(guiDlg);
    } // setupUi

    void retranslateUi(QWidget *guiDlg)
    {
        guiDlg->setWindowTitle(QApplication::translate("guiDlg", "Social Navigation Dataset Generator", Q_NULLPTR));
        configuration->setText(QApplication::translate("guiDlg", "configuration", Q_NULLPTR));
        regenerate->setText(QApplication::translate("guiDlg", "regenerate", Q_NULLPTR));
        label_2->setText(QApplication::translate("guiDlg", "contributor's unique id:", Q_NULLPTR));
        contributor->setText(QApplication::translate("guiDlg", "default", Q_NULLPTR));
        quit->setText(QApplication::translate("guiDlg", "quit", Q_NULLPTR));
        label->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class guiDlg: public Ui_guiDlg {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINUI_H
